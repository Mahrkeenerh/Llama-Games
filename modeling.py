import json
import os
from typing import Union

import peft
import torch
import transformers


def load_VED_vit(model_path, image_size, device):
    """Load the ViT model from a VED model."""

    ved_model = transformers.VisionEncoderDecoderModel.from_pretrained(
        model_path,
        device_map=device
    )

    # delete the language head
    del ved_model.decoder
    torch.cuda.empty_cache()

    vit_processor = transformers.ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_processor.size = {"width": image_size, "height": image_size}

    return model_path, ved_model.encoder, vit_processor


# https://github.com/unslothai/unsloth/commit/ec19e61c854dcf9104386fa63fc6c4f2944d4f35#diff-4c87be791e40a4afa9f8b04a9169460c5ef851be73de2f006898240cd3a43936R480
def fix_untrained_tokens(model, eps = 1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    embedding_matrix = model.get_input_embeddings ().weight.data
    lm_head_matrix   = model.get_output_embeddings().weight.data

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained

    # First set untrained to all 0s - sometimes it's not! 1e-23 for bfloat16
    embedding_matrix[where_untrained] = 0
    lm_head_matrix  [where_untrained] = 0

    # Find sum
    sum_embedding  = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head    = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained).to(embedding_matrix.dtype)
    mean_lm_head   = (sum_lm_head   / n_trained).to(lm_head_matrix  .dtype)

    # Set them to the mean
    embedding_matrix[where_untrained] = mean_embedding
    lm_head_matrix  [where_untrained] = mean_lm_head

    return mean_embedding, mean_lm_head


class LlamaGameDesc(torch.nn.Module):
    def __init__(
        self,
        image_size=448,
        image_merge_factor=4,
        vit_load_func=None,
        llama_id="meta-llama/Meta-Llama-3-8B",
        tokenizer_id="tokenizer",
        lora:Union[str, peft.LoraConfig]=None,
        projector_path=None,
        device=None
    ):
        assert device is not None, "Device must be provided."
        super(LlamaGameDesc, self).__init__()

        self.image_size = image_size
        self.image_merge_factor = image_merge_factor
        self.image_embed_count = image_size ** 2 // 16 ** 2 // image_merge_factor
        self.device = device
        self.is_distributed = False

        self._init_vit(vit_load_func)
        self._init_llama(llama_id, tokenizer_id, lora)
        self._init_embed_projector(projector_path)

    def _init_vit(self, load_func):
        """Apply custom modifications to the ViT model
        - Double input image size
        """

        if load_func is None:
            self.vit_path = None
            self.vit = None
            self.vit_processor = None
            return
        else:
            self.vit_path, self.vit, self.vit_processor = load_func()

        self.vit.embeddings.patch_embeddings.image_size = [self.image_size, self.image_size]

        # Upscale position embeddings
        if self.image_size != 224:
            old_position_embeddings = self.vit.embeddings.position_embeddings
            first_row = old_position_embeddings[:, 0, :].clone().unsqueeze(0)
            rest_upscaled = torch.nn.functional.interpolate(
                old_position_embeddings[:, 1:, :].clone().unsqueeze(0),
                size=((self.image_size // 16) ** 2, self.vit.embeddings.patch_embeddings.projection.out_channels),
                mode="nearest"
            )[0]

            new_position_embeddings = torch.nn.Parameter(
                torch.cat((first_row, rest_upscaled), dim=1),
                requires_grad=True
            ).to(self.device)

            self.vit.embeddings.position_embeddings = new_position_embeddings

        # Disable unused layer
        self.vit.pooler = None

    def _init_llama(self, llama_id, tokenizer_id, lora):
        """Apply custom modifications to the Llama model
        - Init or load LoRA if provided
        - Add special tokens for image embedding
        """

        self.is_lora = lora is not None
        self.llama = transformers.LlamaForCausalLM.from_pretrained(
            llama_id,
            device_map=self.device
        )
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            tokenizer_id,
            padding_side="right"
        )

        # Fix untrained tokens
        fix_untrained_tokens(self.llama)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if isinstance(lora, str):
            self.llama = peft.PeftModel.from_pretrained(
                self.llama,
                os.path.abspath(lora)
            )

        else:
            llama = peft.prepare_model_for_kbit_training(self.llama)
            self.llama = peft.get_peft_model(llama, lora)

        llama_embeddings = self.llama.get_input_embeddings()

        self.bos_embed = llama_embeddings(torch.tensor(self.tokenizer.bos_token_id, device=self.device))
        self.eos_embed = llama_embeddings(torch.tensor(self.tokenizer.eos_token_id, device=self.device))

        self.begin_img_token = self.tokenizer("<|begin_img|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.end_img_token = self.tokenizer("<|end_img|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.begin_img_embed = llama_embeddings(self.begin_img_token)
        self.end_img_embed = llama_embeddings(self.end_img_token)

        self.desc_token = self.tokenizer("<|game_description|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.desc_embed = llama_embeddings(self.desc_token)

    def _init_embed_projector(self, projector_path):
        """Initialize the projector from ViT to Llama embeds."""

        image_embed_size = self.vit.embeddings.patch_embeddings.projection.out_channels
        word_embed_size = self.llama.get_input_embeddings().embedding_dim

        self.embed_projector = torch.nn.Linear(
            image_embed_size * self.image_merge_factor,
            word_embed_size,
            device=self.device
        )

        if projector_path is not None:
            self._load_projector(projector_path)

    @classmethod
    def from_pretrained(
        self,
        path,
        device=None
    ):
        """Load the model from a pretrained model,
        initialize based on config."""

        device = device or torch.device("cuda")

        with open(os.path.join(path, "llama_game_desc_config.json"), "r") as f:
            config = json.load(f)

        if config["vit_path"] is None:
            vit_load_func = None
        else:
            vit_load_func = lambda: load_VED_vit(
                model_path=config["vit_path"],
                device=device
            )

        is_lora = config["lora"]
        lora = os.path.join(path, "lora") if is_lora else None

        llama_id = config["llama_id"]
        if llama_id == "local_llama":
            llama_id = os.path.join(path, "llama")

        tokenizer_id = os.path.join(path, "tokenizer")

        model = LlamaGameDesc(
            image_size=config["image_size"],
            vit_load_func=vit_load_func,
            llama_id=llama_id,
            tokenizer_id=tokenizer_id,
            lora=lora,
            projector_path=os.path.join(path, "projector"),
            device=device
        )

        return model

    def save_pretrained(self, path, merge_lora=False, local_rank=0):
        """Save the model to a specified path."""

        if self.is_distributed and local_rank != 0:
            return

        if merge_lora:
            assert self.is_lora, "Model is not Lora, but merge_lora is True."

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "llama_game_desc_config.json"), "w") as f:
            json.dump({
                "vit_path": self.vit_path,
                "image_size": self.image_size,
                "llama_id": "meta-llama/Meta-Llama-3-8B" if not merge_lora else "local_llama",
                "lora": self.is_lora and not merge_lora
            }, f, indent=4)

        self._save_projector(os.path.join(path, "projector"))

        if self.is_lora:
            if merge_lora:
                self.llama.merge_and_unload(os.path.join(path, "llama"))
            else:
                self.llama.save_pretrained(os.path.join(path, "lora"))

        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

    def eval(self):
        """Set the model to evaluation mode."""

        self.vit.eval()
        self.llama.eval()

        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.embed_projector.parameters():
            param.requires_grad = False

        for param in self.llama.parameters():
            param.requires_grad = False

    def _save_projector(self, path):
        if self.is_distributed:
            projector = self.embed_projector.module
        else:
            projector = self.embed_projector

        torch.save(projector.state_dict(), path)

    def _load_projector(self, path):
        if self.is_distributed:
            projector = self.embed_projector.module
        else:
            projector = self.embed_projector

        projector.load_state_dict(torch.load(path))

    def set_optimizer(self, optim, lr):
        """Set the optimizer for model."""

        self.optimizer = optim(self.parameters(), lr=lr)

        return self.optimizer

    def distribute(self, local_rank):
        self.embed_projector = torch.nn.parallel.DistributedDataParallel(
            self.embed_projector,
            device_ids=[local_rank],
            output_device=local_rank
        )
        self.is_distributed = True
        self.local_rank = local_rank

    def print_parameters(self):
        """Print the number of parameters in the model."""

        if self.is_distributed and self.local_rank != 0:
            return

        if self.vit is None:
            print("ViT model is not provided.")
        else:
            vit_params = sum(p.numel() for p in self.vit.parameters())
            vit_trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            print(f"ViT trainable params: {vit_trainable_params:,} || all params: {vit_params:,} || trainable/all: {vit_trainable_params / vit_params * 100:.2f}%")

        projector_params = sum(p.numel() for p in self.embed_projector.parameters())
        projector_trainable_params = sum(p.numel() for p in self.embed_projector.parameters() if p.requires_grad)
        print(f"Projector trainable params: {projector_trainable_params:,} || all params: {projector_params:,} || trainable/all: {projector_trainable_params / projector_params * 100:.2f}%")

        llama_params = sum(p.numel() for p in self.llama.parameters())
        llama_trainable_params = sum(p.numel() for p in self.llama.parameters() if p.requires_grad)
        print(f"Llama trainable params: {llama_trainable_params:,} || all params: {llama_params:,} || trainable/all: {llama_trainable_params / llama_params * 100:.2f}%")

    def _prepare_image_encoding(self, image_batches):
        """Prepare image encodings from the ViT model."""

        enc_outs = []
        for image_batch in image_batches:
            image_embeds = self.vit(image_batch).last_hidden_state
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(
                bs,
                int(pn / self.image_merge_factor),
                int(hs * self.image_merge_factor)
            )

            enc_outs.append(image_embeds)

        return torch.cat(enc_outs, dim=1)

    def _prepare_llama_input(self, project_embed, labels):
        """Concat project_embeds with labels, prepare input_ids, attentions."""

        label_embeds = self.llama.get_input_embeddings()(labels)

        project_embed_split = torch.split(project_embed, self.image_embed_count, dim=1)

        # build image input: <|begin_img|> + image + <|end_img|>
        project_embed = torch.cat([
            torch.cat((
                self.begin_img_embed.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach(),
                image_embed,
                self.end_img_embed.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach()
            ), dim=1) for image_embed in project_embed_split
        ], dim=1)

        # <|game_description|> + images + label
        inputs_embeds = torch.cat((
            self.desc_embed.unsqueeze(0).repeat(project_embed.size(0), 1, 1),
            project_embed,
            label_embeds
        ), dim=1)

        attentions = torch.ones(inputs_embeds.size()[:-1], device=self.device)

        label_ids = torch.cat((
            torch.full(
                (torch.tensor(project_embed.size()[:-1]) + torch.tensor([0, 1])).tolist(),
                -100,
                device=self.device
            ),
            labels
        ), dim=1)

        return inputs_embeds, attentions, label_ids

    def forward(
        self,
        image_batches = None,
        labels = None,
        encoder_outputs = None
    ):
        assert image_batches is not None or encoder_outputs is not None, "Either image_batches or encoder_outputs must be provided."
        assert image_batches is None or encoder_outputs is None, "Only one of image_batches or encoder_outputs can be provided."

        if encoder_outputs is None:
            encoder_outputs = self._prepare_image_encoding(image_batches)

        projected_embed = self.embed_projector(encoder_outputs)

        inputs_embeds, attentions, label_ids = self._prepare_llama_input(projected_embed, labels)

        llama_out = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            labels=label_ids
        )

        return llama_out

    def generate(
        self,
        image_batches=None,
        encoder_outputs=None,
        max_new_tokens=512,
        do_decode=True
    ):
        assert image_batches is not None or encoder_outputs is not None, "Either image_batches or encoder_outputs must be provided."
        assert image_batches is None or encoder_outputs is None, "Only one of image_batches or encoder_outputs can be provided."

        if encoder_outputs is None:
            encoder_outputs = self._prepare_image_encoding(image_batches)

        projected_embed = self.embed_projector(encoder_outputs)

        inputs_embeds, attentions, label_ids = self._prepare_llama_input(projected_embed, torch.tensor([[]], device=self.device, dtype=torch.long))

        generated_ids = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
        )[0]

        if do_decode:
            out = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return out

        return generated_ids
