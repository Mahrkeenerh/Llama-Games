import json
import os
from typing import Self

import peft
from peft import LoraConfig, PeftModel
import torch
from transformers import ViTModel, ViTImageProcessor, LlamaForSequenceClassification, LlamaForCausalLM, PreTrainedTokenizerFast


class Projector(torch.nn.Module):
    def __init__(
        self,
        image_embed_size=768,
        image_merge_factor=4,
        word_embed_size=4096,
        device=None
    ):
        super(Projector, self).__init__()

        self.device = device or torch.device("cuda")
        self.image_merge_factor = image_merge_factor
        self.image_embed_size = image_embed_size
        self.word_embed_size = word_embed_size

        self.projector = torch.nn.Linear(
            image_embed_size * self.image_merge_factor,
            word_embed_size,
            device=self.device
        )

    @classmethod
    def from_pretrained(self, path, device=None) -> Self:
        """Load the projector from a specified path."""

        with open(os.path.join(path, "projector_config.json"), "r") as f:
            config = json.load(f)

        model = Projector(
            image_embed_size=config["image_embed_size"],
            image_merge_factor=config["image_merge_factor"],
            word_embed_size=config["word_embed_size"],
            device=device
        )

        model.projector.load_state_dict(torch.load(os.path.join(path, "model.pt")))

        return model

    def save_pretrained(self, path):
        """Save the projector to a specified path."""

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "projector_config.json"), "w") as f:
            json.dump({
                "image_embed_size": self.image_embed_size,
                "image_merge_factor": self.image_merge_factor,
                "word_embed_size": self.word_embed_size
            }, f, indent=4)
        torch.save(self.projector.state_dict(), os.path.join(path, "model.pt"))

    def forward(self, image_embeds):
        return self.projector(image_embeds)


class LlamaGameBase(torch.nn.Module):
    def __init__(
        self,
        device=None
    ):
        super(LlamaGameBase, self).__init__()

        self.device = device or torch.device("cuda")
        self.is_distributed = False

        self.vit = None
        self.projector = None
        self.llama = None

    def load_vit(self, path):
        """Load ViT model from a specified path."""

        self.vit = ViTModel.from_pretrained(path, device_map=self.device)
        self.vit_processor = ViTImageProcessor.from_pretrained(path, device=self.device)

        self.vit.pooler = None
        torch.cuda.empty_cache()

    def load_projector(self, path):
        """Load projector model from a specified path."""

        self.projector = Projector.from_pretrained(path, device=self.device)

    def _save_vit(self, path):
        """Save ViT model to a specified path."""

        if self.is_distributed and self.local_rank != 0:
            return

        vit = self.vit.module if self.is_distributed else self.vit
        vit.save_pretrained(path)
        self.vit_processor.save_pretrained(path)

    def _save_projector(self, path):
        """Save projector model to a specified path."""

        if self.is_distributed and self.local_rank != 0:
            return

        projector = self.projector.module if self.is_distributed else self.projector
        projector.save_pretrained(path)

    def print_parameters(self):
        """Print the total number of parameters and trainable parameters in the model."""

        if self.is_distributed and self.local_rank != 0:
            return

        if self.vit is None:
            print("ViT model is not loaded.")
        else:
            vit_params = sum(p.numel() for p in self.vit.parameters())
            vit_trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            print(f"ViT trainable params: {vit_trainable_params:,} || all params: {vit_params:,} || trainable/all: {vit_trainable_params / vit_params * 100:.2f}%")

        if self.projector is None:
            print("Projector model is not loaded.")
        else:
            projector_params = sum(p.numel() for p in self.projector.parameters())
            projector_trainable_params = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
            print(f"Projector trainable params: {projector_trainable_params:,} || all params: {projector_params:,} || trainable/all: {projector_trainable_params / projector_params * 100:.2f}%")

        if self.llama is None:
            print("Llama model is not loaded.")
        else:
            llama_params = sum(p.numel() for p in self.llama.parameters())
            llama_trainable_params = sum(p.numel() for p in self.llama.parameters() if p.requires_grad)
            print(f"Llama trainable params: {llama_trainable_params:,} || all params: {llama_params:,} || trainable/all: {llama_trainable_params / llama_params * 100:.2f}%")

    def distribute(self, local_rank):
        """Distribute the model across GPUs."""

        if self.vit is not None:
            self.vit = torch.nn.parallel.DistributedDataParallel(
                self.vit,
                device_ids=[local_rank],
                output_device=local_rank
            )

        if self.projector is not None:
            self.projector = torch.nn.parallel.DistributedDataParallel(
                self.projector,
                device_ids=[local_rank],
                output_device=local_rank
            )

        if self.llama is not None:
            self.llama = torch.nn.parallel.DistributedDataParallel(
                self.llama,
                device_ids=[local_rank],
                output_device=local_rank
            )

        self.is_distributed = True
        self.local_rank = local_rank

    def train(self, modules=None):
        """Set specified model modules to training mode."""

        if modules is None:
            modules = [self.vit, self.llama, self.projector]

        for module in modules:
            if module is not None:
                module.train()

    def eval(self):
        """Set the model to evaluation mode."""

        if self.vit is not None:
            self.vit.eval()

        if self.llama is not None:
            self.llama.eval()

    def get_image_encoding(self, image_batches):
        """
        Prepare image encodings from the ViT model.
        Merge the image embeddings based on the image_merge_factor.
        Concatenate image_batches to a single tensor.
        """

        enc_outs = []
        for image_batch in image_batches:
            image_embeds = self.vit(image_batch).last_hidden_state
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(
                bs,
                int(pn / self.projector.image_merge_factor),
                int(hs * self.projector.image_merge_factor)
            )

            enc_outs.append(image_embeds)

        return torch.cat(enc_outs, dim=1)

    def get_projector_outputs(self, image_batches, encoder_outputs, projector_outputs):
        """
        Get the projector outputs:
        image_batches -> encoder_outputs -> projector_outputs
        """

        assert image_batches is not None or encoder_outputs is not None or projector_outputs is not None, "Either image_batches or encoder_outputs or projector_outputs must be provided."
        assert image_batches is None or encoder_outputs is None or projector_outputs is None, "Only one of image_batches or encoder_outputs or projector_outputs can be provided."

        if projector_outputs is None:
            if encoder_outputs is None:
                encoder_outputs = self.get_image_encoding(image_batches)

            projector_outputs = self.projector(encoder_outputs)

        return projector_outputs


class LlamaGameClassification(LlamaGameBase):
    def __init__(
        self,
        num_labels=128,
        projector_config=None,
        device=None
    ):
        super(LlamaGameClassification, self).__init__(device=device)

        self.num_labels = num_labels

        if projector_config is not None:
            self.projector = Projector(
                **projector_config,
                device=self.device
            )

        self.llama = LlamaForSequenceClassification.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            num_labels=self.num_labels,
            device_map=self.device
        )

        # Freeze the llama model
        for param in self.llama.model.parameters():
            param.requires_grad = False

    def _save_classificator(self, path):
        """Save the classificator to a specified path."""

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "classificator_config.json"), "w") as f:
            json.dump({
                "num_labels": self.num_labels
            }, f, indent=4)
        torch.save(self.llama.score.state_dict(), os.path.join(path, "model.pt"))

    def _load_classificator(self, path):
        """Load the classificator from a specified path."""

        self.llama.score.load_state_dict(torch.load(
            os.path.join(path, "model.pt")
        ))

    @classmethod
    def from_pretrained(
        self,
        path,
        num_labels=None,
        load_modules=[],
        device=None
    ) -> Self:
        """
        Load the model from a pretrained model,
        initialize based on config.

        load_modules: list of modules to load, if None, load all.
            possible values: ["vit", "projector", "llama.score"]
        """

        device = device or torch.device("cuda")

        assert num_labels is None or "llama.score" not in load_modules, 'num_labels must not be provided if "llama.score" is in load_modules.'

        if not load_modules:
            load_modules = ["vit", "projector", "llama.score"]

        for module in load_modules:
            assert module in ["vit", "projector", "llama.score"], f"Invalid module: {module}."

        assert num_labels is not None or "llama.score" in load_modules, 'num_labels must be provided if "llama.score" is not in load_modules.'

        if num_labels is None or "llama.score" in load_modules:
            classificator_path = os.path.join(path, "classificator")
            with open(os.path.join(classificator_path, "classificator_config.json"), "r") as f:
                config = json.load(f)
                saved_num_labels = config["num_labels"]

        model = LlamaGameClassification(
            num_labels=num_labels or saved_num_labels,
            device=device
        )

        if "vit" in load_modules:
            if not os.path.exists(os.path.join(path, "vit")):
                print("ViT model is not loaded.")
            else:
                model.load_vit(os.path.join(path, "vit"))

        if "projector" in load_modules:
            if not os.path.exists(os.path.join(path, "projector")):
                print("Projector model is not loaded.")
            else:
                model.load_projector(os.path.join(path, "projector"))

        # If num_labels is provided, do not load the classificator
        if "llama.score" in load_modules and num_labels is None:
            model._load_classificator(os.path.join(path, "classificator"))

        return model

    def save_pretrained(self, path, save_modules=None):
        """
        Save the model to a specified path.

        save_modules: list of modules to save, if None, save all.
            possible values: ["vit", "projector", "llama.score"]
        """

        if self.is_distributed and self.local_rank != 0:
            return

        assert save_modules is None or type(save_modules) == list, "save_modules must be None or a list."

        if save_modules is None:
            save_modules = ["vit", "projector", "llama.score"]

        os.makedirs(path, exist_ok=True)

        if self.vit is not None and "vit" in save_modules:
            self._save_vit(os.path.join(path, "vit"))

        if self.projector is not None and "projector" in save_modules:
            self._save_projector(os.path.join(path, "projector"))

        if "llama.score" in save_modules:
            self._save_classificator(os.path.join(path, "classificator"))

    def forward(
        self,
        image_batches = None,
        labels = None,
        encoder_outputs = None,
        projector_outputs = None
    ):

        projector_outputs = self.get_projector_outputs(image_batches, encoder_outputs, projector_outputs)

        llama_out = self.llama(
            input_ids=projector_outputs,
            labels=labels
        )

        return llama_out


class LlamaGameDescription(LlamaGameBase):
    def __init__(
        self,
        tokenizer_path="models/tokenizer",
        lora_config:LoraConfig=None,
        device=None
    ):
        super(LlamaGameDescription, self).__init__(device=device)

        self.llama = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            device_map=self.device
        )
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path,
            padding_side="right"
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._fix_untrained_tokens()

        llama_embeddings = self.llama.get_input_embeddings()

        self.bos_embed = llama_embeddings(torch.tensor(self.tokenizer.bos_token_id, device=self.device))
        self.eos_embed = llama_embeddings(torch.tensor(self.tokenizer.eos_token_id, device=self.device))

        self.begin_img_token = self.tokenizer("<|begin_img|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.end_img_token = self.tokenizer("<|end_img|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.begin_img_embed = llama_embeddings(self.begin_img_token)
        self.end_img_embed = llama_embeddings(self.end_img_token)

        self.desc_token = self.tokenizer("<|game_description|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.desc_embed = llama_embeddings(self.desc_token)

        if lora_config is not None:
            self.init_lora(lora_config)

    def init_lora(self, lora_config):
        """Initialize LoRA model."""
        self.is_lora = True

        self.llama = peft.prepare_model_for_kbit_training(self.llama)
        self.llama = peft.get_peft_model(self.llama, lora_config)

    def _load_lora(self, path):
        """Load LoRA model from a specified path."""

        self.llama = PeftModel.from_pretrained(
            self.llama,
            path
        )

    def _save_lora(self, path):
        """Save LoRA model to a specified path."""

        if self.is_distributed and self.local_rank != 0:
            return

        llama = self.llama.module if self.is_distributed else self.llama
        llama.save_pretrained(path)

    # https://github.com/unslothai/unsloth/commit/ec19e61c854dcf9104386fa63fc6c4f2944d4f35#diff-4c87be791e40a4afa9f8b04a9169460c5ef851be73de2f006898240cd3a43936R480
    def _fix_untrained_tokens(self, eps = 1e-16):
        """
        Llama-3 for eg has untrained vectors in the base model.
        These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
        We reset them to the mean of the rest of the tokens
        """
        embedding_matrix = self.llama.get_input_embeddings ().weight.data
        lm_head_matrix   = self.llama.get_output_embeddings().weight.data

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

    @classmethod
    def from_pretrained(
        self,
        path,
        load_modules=[],
        device=None
    ) -> Self:
        """
        Load the model from a pretrained model,
        initialize based on config.

        load_modules: list of modules to load, if None, load all.
            possible values: ["vit", "projector", "lora"]
        """

        device = device or torch.device("cuda")

        if not load_modules:
            load_modules = ["vit", "projector", "lora"]

        for module in load_modules:
            assert module in ["vit", "projector", "lora"], f"Invalid module: {module}."

        model = LlamaGameDescription(
            tokenizer_path=os.path.join(path, "tokenizer"),
            device=device
        )

        if "vit" in load_modules:
            if not os.path.exists(os.path.join(path, "vit")):
                print("ViT model is not loaded.")
            else:
                model.load_vit(os.path.join(path, "vit"))

        if "projector" in load_modules:
            if not os.path.exists(os.path.join(path, "projector")):
                print("Projector model is not loaded.")
            else:
                model.load_projector(os.path.join(path, "projector"))

        if "lora" in load_modules:
            if not os.path.exists(os.path.join(path, "lora")):
                print("LoRA model is not loaded.")
            else:
                model._load_lora(os.path.join(path, "lora"))

        return model

    def save_pretrained(self, path, save_modules=None):
        """
        Save the model to a specified path.

        save_modules: list of modules to save, if None, save all.
            possible values: ["vit", "projector", "lora"]
        """

        if self.is_distributed and self.local_rank != 0:
            return

        assert save_modules is None or type(save_modules) == list, "save_modules must be None or a list."

        if save_modules is None:
            save_modules = ["vit", "projector", "lora"]

        for module in save_modules:
            assert module in ["vit", "projector", "lora"], f"Invalid module: {module}."

        if "lora" in save_modules:
            assert self.is_lora, "Model is not LoRA but 'lora' is in save_modules."

        os.makedirs(path, exist_ok=True)

        if self.vit is not None and "vit" in save_modules:
            self._save_vit(os.path.join(path, "vit"))

        if self.projector is not None and "projector" in save_modules:
            self._save_projector(os.path.join(path, "projector"))

        if self.is_lora and "lora" in save_modules:
            self._save_lora(os.path.join(path, "lora"))

        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

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
        encoder_outputs = None,
        projector_outputs = None
    ):
        projector_outputs = self.get_projector_outputs(image_batches, encoder_outputs, projector_outputs)
        inputs_embeds, attentions, label_ids = self._prepare_llama_input(projector_outputs, labels)

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
        projector_outputs=None,
        max_new_tokens=512,
        do_decode=True
    ):
        projector_outputs = self.get_projector_outputs(image_batches, encoder_outputs, projector_outputs)

        inputs_embeds, attentions, label_ids = self._prepare_llama_input(projector_outputs, torch.tensor([[]], device=self.device, dtype=torch.long))

        generated = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
        )

        if do_decode:
            out = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return out

        return generated
