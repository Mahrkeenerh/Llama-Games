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


def load_mixtral(model_path, load_4bit, device):
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        model_path,
        padding_side="right",
        device_map=device
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        attn_implementation = "flash_attention_2"
    else:
        quantization_config = None
        attn_implementation = None

    mixtral = transformers.MixtralForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        device_map=device
    )

    return model_path, mixtral, tokenizer, load_4bit


def load_mistral(model_path, load_4bit, device, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path

    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        tokenizer_path,
        padding_side="right",
        device_map=device
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        attn_implementation = "flash_attention_2"
    else:
        quantization_config = None
        attn_implementation = None

    mistral = transformers.MistralForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        device_map=device
    )

    return model_path, mistral, tokenizer, load_4bit


class Vixtral(torch.nn.Module):
    def __init__(
        self,
        image_size=448,
        image_merge_factor=4,
        vit_load_func=None,
        mixtral_load_func=None,
        lora:Union[str, peft.LoraConfig]=None,
        projector_path=None,
        device=None
    ):
        assert device is not None, "Device must be provided."
        assert mixtral_load_func is not None, "Mixtral model must be provided."
        super(Vixtral, self).__init__()

        self.image_size = image_size
        self.image_merge_factor = image_merge_factor
        self.image_embed_count = image_size ** 2 // 16 ** 2 // image_merge_factor
        self.device = device
        self.is_distributed = False

        self._init_vit(vit_load_func)
        self._init_mixtral(mixtral_load_func, lora)
        self._init_embed_projector(projector_path)

    def _init_vit(self, load_func):
        """Apply custom modifications to the ViT model
        - Double input image size
        - Freeze the whole model
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

        # Freeze the whole model
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.eval()

    def _init_mixtral(self, load_func, lora):
        """Apply custom modifications to the Mixtral model
        - Init or load LoRA if provided
        """

        self.mixtral_path, self.mixtral, self.tokenizer, self.quantized = load_func()

        self.bos_embed = self.mixtral.get_input_embeddings()(torch.tensor(self.tokenizer.bos_token_id, device=self.device))
        self.eos_embed = self.mixtral.get_input_embeddings()(torch.tensor(self.tokenizer.eos_token_id, device=self.device))

        self.image_prepend_tokens = self.tokenizer("<img>", return_tensors="pt").input_ids[0][1:].to(self.device)
        self.image_append_tokens = self.tokenizer("</img>", return_tensors="pt").input_ids[0][1:].to(self.device)
        self.image_prepend_embeds = self.mixtral.get_input_embeddings()(self.image_prepend_tokens)
        self.image_append_embeds = self.mixtral.get_input_embeddings()(self.image_append_tokens)

        if lora is None:
            self.lora_mixtral = None

        elif isinstance(lora, str):
            self.lora_mixtral = peft.PeftModel.from_pretrained(
                self.mixtral,
                os.path.abspath(lora)
            )

        else:
            mixtral = peft.prepare_model_for_kbit_training(self.mixtral)
            self.lora_mixtral = peft.get_peft_model(mixtral, lora)

    def _init_embed_projector(self, projector_path):
        """Initialize the projector from ViT to Mixtral embeds."""

        image_embed_size = self.vit.embeddings.patch_embeddings.projection.out_channels
        word_embed_size = self.mixtral.get_input_embeddings().embedding_dim

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
        load_4bit=False,
        device=None
    ):
        """Load the ViXtral model from a pretrained model,
        initialize based on config."""

        device = device or torch.device("cuda")

        with open(os.path.join(path, "vixtral_config.json"), "r") as f:
            config = json.load(f)

        if config["vit_path"] is None:
            vit_load_func = None
        else:
            vit_load_func = lambda: load_VED_vit(
                model_path=config["vit_path"],
                device=device
            )

        mixtral_load_func = lambda: load_mixtral(
            model_path=config["mixtral_path"],
            load_4bit=load_4bit,
            device=device
        )

        is_lora = os.path.exists(os.path.join(path, "lora"))
        lora = os.path.join(path, "lora") if is_lora else None

        vixtral = Vixtral(
            image_size=config["image_size"],
            vit_load_func=vit_load_func,
            mixtral_load_func=mixtral_load_func,
            lora=lora,
            projector_path=os.path.join(path, "projector"),
            device=device
        )

        return vixtral

    def save_pretrained(self, path, merge_lora=False, local_rank=0):
        """Save the ViXtral model to a path."""

        if self.is_distributed and local_rank != 0:
            return

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "vixtral_config.json"), "w") as f:
            json.dump({
                "vit_path": self.vit_path,
                "mixtral_path": self.mixtral_path,
                "image_size": self.image_size,
                "lora": not merge_lora
            }, f, indent=4)

        self._save_projector(os.path.join(path, "projector"))

        if self.lora_mixtral is not None:
            if merge_lora:
                self.lora_mixtral.merge_and_unload(os.path.join(path, "mixtral_lora"))
            else:
                self.lora_mixtral.save_pretrained(os.path.join(path, "lora"))

    def eval(self):
        """Set the ViXtral model to evaluation mode."""

        if self.lora_mixtral is not None:
            self.lora_mixtral.eval()
        else:
            self.mixtral.eval()

        for param in self.embed_projector.parameters():
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
        """Set the optimizer for the ViXtral model."""

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
        """Print the number of parameters in the ViXtral model."""

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

        mixtral_params = sum(p.numel() for p in self.mixtral.parameters())
        mixtral_trainable_params = sum(p.numel() for p in self.mixtral.parameters() if p.requires_grad)
        print(f"Mixtral trainable params: {mixtral_trainable_params:,} || all params: {mixtral_params:,} || trainable/all: {mixtral_trainable_params / mixtral_params * 100:.2f}%")

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

    def _prepare_mixtral_input(self, project_embed, labels):
        """Concat project_embeds with labels, prepare input_ids, attentions."""

        label_embeds = self.mixtral.get_input_embeddings()(labels)

        project_embed_split = torch.split(project_embed, self.image_embed_count, dim=1)

        # build image input: <img> + image + </img>
        project_embed = torch.cat([
            torch.cat((
                self.image_prepend_embeds.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach(),
                image_embed,
                self.image_append_embeds.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach()
            ), dim=1) for image_embed in project_embed_split
        ], dim=1)

        # <bos> + images + label
        inputs_embeds = torch.cat((
            self.bos_embed.unsqueeze(0).repeat(project_embed.size(0), 1, 1).detach(),
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

        dtype = torch.float16 if self.quantized else torch.float32
        projected_embed = self.embed_projector(encoder_outputs).type(dtype)

        inputs_embeds, attentions, label_ids = self._prepare_mixtral_input(projected_embed, labels)

        mixtral_out = self.mixtral(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            labels=label_ids
        )

        return mixtral_out

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

        dtype = torch.float16 if self.quantized else torch.float32
        projected_embed = self.embed_projector(encoder_outputs).type(dtype)

        inputs_embeds, attentions, label_ids = self._prepare_mixtral_input(projected_embed, torch.tensor([[]], device=self.device, dtype=torch.long))

        generated_ids = self.mixtral.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
        )[0]

        if do_decode:
            out = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return out

        return generated_ids
