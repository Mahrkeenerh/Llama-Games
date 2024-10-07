import json
import os
from typing import Self

import peft
from peft import LoraConfig, PeftModel
import torch
from transformers import ViTModel, ViTImageProcessor, LlamaForSequenceClassification, LlamaForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig


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


class LlamaBase(torch.nn.Module):
    def __init__(
        self,
        projector_config=None,
        device=None
    ):
        super(LlamaBase, self).__init__()

        self.device = device or torch.device("cuda")
        self.is_distributed = False
        self.image_embed_count = None
        self.quantized = False

        self.frozen = set()
        self.vit_dist = None
        self.projector_dist = None
        self.llama_dist = None
        self.vit = None
        self.llama = None

        if projector_config is not None:
            self.projector = Projector(
                **projector_config,
                device=self.device
            )
        else:
            self.projector = None

    def load_vit(self, path):
        """Load ViT model from a specified path."""

        self.vit = ViTModel.from_pretrained(path, device_map=self.device)
        self.vit_processor = ViTImageProcessor.from_pretrained(path)

        self.vit.pooler = None
        torch.cuda.empty_cache()

    def load_projector(self, path):
        """Load projector model from a specified path."""

        self.projector = Projector.from_pretrained(path, device=self.device)

    def _save_vit(self, path):
        """Save ViT model to a specified path."""

        if self.is_distributed and self.local_rank != 0:
            return

        self.vit.save_pretrained(path)
        self.vit_processor.save_pretrained(path)

    def _save_projector(self, path):
        """Save projector model to a specified path."""

        if self.is_distributed and self.local_rank != 0:
            return

        self.projector.save_pretrained(path)

    def _get_image_embed_count(self):
        if self.image_embed_count is None:
            if self.projector is None and self.vit is None:
                print("image_embed_count is None, using default value for image size 448 with merge factor of 4.")
                self.image_embed_count = 448 ** 2 // 16 ** 2 // 4
            else:
                self.image_embed_count = self.vit.config.image_size ** 2 // 16 ** 2 // self.projector.image_merge_factor

        return self.image_embed_count

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

        if self.vit is not None and self.vit not in self.frozen:
            self.vit_dist = torch.nn.parallel.DistributedDataParallel(
                self.vit,
                device_ids=[local_rank],
                output_device=local_rank
            )

        if self.projector is not None and self.projector not in self.frozen:
            self.projector_dist = torch.nn.parallel.DistributedDataParallel(
                self.projector,
                device_ids=[local_rank],
                output_device=local_rank
            )

        if self.llama is not None and self.llama not in self.frozen:
            self.llama_dist = torch.nn.parallel.DistributedDataParallel(
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

    def freeze(self, *modules):
        """Freeze specified model modules."""

        if modules is None:
            modules = [self.vit, self.llama, self.projector]

        if not any([isinstance(modules, list), isinstance(modules, tuple)]):
            modules = [modules]

        for module in modules:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                self.frozen.add(module)

    def get_image_encoding(self, image_batches):
        """
        Prepare image encodings from the ViT model.
        Merge the image embeddings based on the image_merge_factor.
        Concatenate image_batches to a single tensor.
        """

        image_merge_factor = self.projector.image_merge_factor
        vit = self.vit if self.vit_dist is None else self.vit_dist

        enc_outs = []
        for image_batch in image_batches:
            image_embeds = vit(image_batch).last_hidden_state
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(
                bs,
                int(pn / image_merge_factor),
                int(hs * image_merge_factor)
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

            projector = self.projector if self.projector_dist is None else self.projector_dist
            projector_outputs = projector(encoder_outputs)

        if self.quantized:
            projector_outputs = projector_outputs.to(torch.float16)

        return projector_outputs


class LlamaGameClassification(LlamaBase):
    def __init__(
        self,
        num_labels=128,
        projector_config=None,
        device=None
    ):
        super(LlamaGameClassification, self).__init__(
            projector_config=projector_config,
            device=device
        )

        self.num_labels = num_labels

        self.llama = LlamaForSequenceClassification.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B",
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

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
            model.image_embed_count = config["image_embed_count"]

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

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "image_embed_count": self.image_embed_count
            }, f, indent=4)

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


class LlamaCaption(LlamaBase):
    def __init__(
        self,
        projector_config=None,
        tokenizer_path="models/tokenizer",
        quantize=False,
        lora_config:LoraConfig=None,
        device=None
    ):
        self.task = "caption"

        super(LlamaCaption, self).__init__(
            projector_config=projector_config,
            device=device
        )

        self.quantized = quantize
        if self.quantized:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            attn_implementation = "flash_attention_2"
        else:
            quant_config = None
            attn_implementation = None

        self.llama = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B",
            quantization_config=quant_config,
            attn_implementation=attn_implementation,
            device_map=self.device
        )
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path,
            padding_side="right"
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.is_lora = False

        if lora_config is not None:
            self.init_lora(lora_config)

        self._init_embeds()
        self._set_embeds()

    def init_lora(self, lora_config):
        """Initialize LoRA model."""

        assert not self.is_lora, "Model already has LoRA."

        self.is_lora = True
        self.llama = peft.get_peft_model(self.llama, lora_config)

    def _load_lora(self, path):
        """Load LoRA model from a specified path."""

        assert not self.is_lora, "Model already has LoRA."

        self.is_lora = True
        self.llama = PeftModel.from_pretrained(
            self.llama,
            path
        )

    def _save_lora(self, path):
        """Save LoRA model to a specified path."""

        if self.is_distributed and self.local_rank != 0:
            return

        self.llama.save_pretrained(path)

    def _init_embeds(self):
        """
        Initialize special tokens embeddings.
        Utilize smart embedding initialization where possible.
        Randomly initialize other tokens.
        """

        embedding_matrix = self.llama.get_input_embeddings().weight.data

        vocab_map = {
            "<|image_caption|>": 128011,
            "<|game_description|>": 128012,
            "<|begin_img|>": 128013,
            "<|end_img|>": 128014,
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "Image": 1945,
            "Text": 1199
        }

        mean_embedding = torch.mean(embedding_matrix, dtype=torch.float32, axis=0)
        var_embedding  = torch.var(embedding_matrix, dim=0)

        # Randomly initialize untrained tokens with mean var
        embedding_matrix[vocab_map["<|image_caption|>"]] = mean_embedding + torch.randn_like(mean_embedding) * var_embedding
        embedding_matrix[vocab_map["<|game_description|>"]] = mean_embedding + torch.randn_like(mean_embedding) * var_embedding

        # Utilize smart embedding initialization with algebraic operations
        embedding_matrix[vocab_map["<|begin_img|>"]] = (
            embedding_matrix[vocab_map["<|begin_of_text|>"]]
            - embedding_matrix[vocab_map["Text"]]
            + embedding_matrix[vocab_map["Image"]]
        )
        embedding_matrix[vocab_map["<|end_img|>"]] = (
            embedding_matrix[vocab_map["<|end_of_text|>"]]
            - embedding_matrix[vocab_map["Text"]]
            + embedding_matrix[vocab_map["Image"]]
        )

    def _set_embeds(self):
        self.llama_embeddings = self.llama.get_input_embeddings()

        self.bos_embed = self.llama_embeddings(torch.tensor(self.tokenizer.bos_token_id, device=self.device))
        self.eos_embed = self.llama_embeddings(torch.tensor(self.tokenizer.eos_token_id, device=self.device))

        self.begin_img_token = self.tokenizer("<|begin_img|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.end_img_token = self.tokenizer("<|end_img|>", return_tensors="pt").input_ids[0][1].to(self.device)

        self.desc_token = self.tokenizer("<|game_description|>", return_tensors="pt").input_ids[0][1].to(self.device)
        self.caption_token = self.tokenizer("<|image_caption|>", return_tensors="pt").input_ids[0][1].to(self.device)

    @property
    def begin_img_embed(self):
        return self.llama.get_input_embeddings()(self.begin_img_token)

    @property
    def end_img_embed(self):
        return self.llama.get_input_embeddings()(self.end_img_token)

    @property
    def task_embed(self):
        if self.task == "caption":
            return self.llama.get_input_embeddings()(self.caption_token)
        elif self.task == "description":
            return self.llama.get_input_embeddings()(self.desc_token)

    @classmethod
    def from_pretrained(
        self,
        path,
        quantize=False,
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

        model = LlamaCaption(
            tokenizer_path=os.path.join(path, "tokenizer"),
            quantize=quantize,
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

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
            model.image_embed_count = config["image_embed_count"]

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
            save_modules = []
            if self.vit is not None:
                save_modules.append("vit")

            if self.projector is not None:
                save_modules.append("projector")

            if self.is_lora:
                save_modules.append("lora")

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

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "image_embed_count": self.image_embed_count
            }, f, indent=4)

    def _prepare_llama_input(self, project_embed, labels):
        """Concat project_embeds with labels, prepare input_ids, attentions."""

        label_embeds = self.llama_embeddings(labels)
        project_embed_split = torch.split(project_embed, self._get_image_embed_count(), dim=1)

        begin_img_embed = self.begin_img_embed
        end_img_embed = self.end_img_embed

        # build image input: <|begin_img|> + image + <|end_img|>
        project_embed = torch.cat([
            torch.cat((
                begin_img_embed.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach(),
                image_embed,
                end_img_embed.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach()
            ), dim=1) for image_embed in project_embed_split
        ], dim=1)

        task_embed = self.task_embed

        # <|bos|> + images + <|task|> + label
        inputs_embeds = torch.cat((
            self.bos_embed.unsqueeze(0).repeat(project_embed.size(0), 1, 1).detach(),
            project_embed,
            task_embed.unsqueeze(0).repeat(project_embed.size(0), 1, 1).detach(),
            label_embeds
        ), dim=1)

        attentions = torch.ones(inputs_embeds.size()[:-1], device=self.device)

        label_ids = torch.cat((
            torch.full(
                (torch.tensor(project_embed.size()[:-1]) + torch.tensor([0, 2])).tolist(),
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
        do_decode=True,
        **generation_kwargs
    ):
        projector_outputs = self.get_projector_outputs(image_batches, encoder_outputs, projector_outputs)

        # create empty labels of shape (bs, 0)
        labels = torch.empty(
            (projector_outputs.size(0), 0),
            dtype=torch.long,
            device=self.device
        )
        inputs_embeds, attentions, label_ids = self._prepare_llama_input(projector_outputs, labels)

        generated = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs
        )

        if do_decode:
            out = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return out

        return generated


class LlamaGameDescription(LlamaCaption):
    def __init__(
        self,
        projector_config=None,
        tokenizer_path="models/tokenizer",
        quantize=False,
        lora_config:LoraConfig=None,
        device=None
    ):
        super(LlamaGameDescription, self).__init__(
            projector_config=projector_config,
            tokenizer_path=tokenizer_path,
            quantize=quantize,
            lora_config=lora_config,
            device=device
        )

        self.task = "description"

        self._init_embeds()
        self._set_embeds()

    @classmethod
    def from_pretrained(
        self,
        path,
        quantize=False,
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
            quantize=quantize,
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

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
            model.image_embed_count = config["image_embed_count"]

        return model

    def _prepare_llama_input(self, project_embed, hints, descriptions):
        """Concat project_embeds with labels, prepare input_ids, attentions."""

        hint_embeds = self.llama_embeddings(hints)
        description_embeds = self.llama_embeddings(descriptions)

        project_embed_split = torch.split(project_embed, self._get_image_embed_count(), dim=1)

        begin_img_embed = self.begin_img_embed
        end_img_embed = self.end_img_embed

        # build image input: <|begin_img|> + image + <|end_img|>
        project_embed = torch.cat([
            torch.cat((
                begin_img_embed.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach(),
                image_embed,
                end_img_embed.unsqueeze(0).repeat(image_embed.shape[0], 1, 1).detach()
            ), dim=1) for image_embed in project_embed_split
        ], dim=1)

        task_embed = self.task_embed

        # <|bos|> + images + <|task|> + hint + description
        inputs_embeds = torch.cat((
            self.bos_embed.unsqueeze(0).repeat(project_embed.size(0), 1, 1).detach(),
            project_embed,
            task_embed.unsqueeze(0).repeat(project_embed.size(0), 1, 1).detach(),
            hint_embeds,
            description_embeds
        ), dim=1)

        attentions = torch.ones(inputs_embeds.size()[:-1], device=self.device)

        label_ids = torch.cat((
            torch.full(
                (torch.tensor(project_embed.size()[:-1]) + torch.tensor([0, 2 + hint_embeds.size()[-2]])).tolist(),
                -100,
                device=self.device
            ),
            descriptions
        ), dim=1)

        return inputs_embeds, attentions, label_ids

    def forward(
        self,
        image_batches = None,
        labels=None,
        encoder_outputs = None,
        projector_outputs = None
    ):
        hints, descriptions = labels

        projector_outputs = self.get_projector_outputs(image_batches, encoder_outputs, projector_outputs)
        inputs_embeds, attentions, label_ids = self._prepare_llama_input(projector_outputs, hints, descriptions)

        llama_out = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            labels=label_ids
        )

        return llama_out

    def generate(
        self,
        image_batches=None,
        hints=None,
        labels=None,
        encoder_outputs=None,
        projector_outputs=None,
        do_decode=True,
        **generation_kwargs
    ):
        assert hints is not None or labels is not None, "Either hints or labels with hints must be provided."

        projector_outputs = self.get_projector_outputs(image_batches, encoder_outputs, projector_outputs)
        hints = hints if hints is not None else labels[0]

        # create empty labels of shape (bs, 0)
        labels = torch.empty(
            (projector_outputs.size(0), 0),
            dtype=torch.long,
            device=self.device
        )

        inputs_embeds, attentions, label_ids = self._prepare_llama_input(projector_outputs, hints, labels)

        generated = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs
        )

        if do_decode:
            out = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return out

        return generated


class LlamaLora(LlamaCaption):
    def __init__(
        self,
        tokenizer_path="models/tokenizer",
        quantize=False,
        lora_config:LoraConfig=None,
        device=None
    ):
        super(LlamaLora, self).__init__(
            projector_config=None,
            tokenizer_path=tokenizer_path,
            quantize=quantize,
            lora_config=lora_config,
            device=device
        )

    @classmethod
    def from_pretrained(
        self,
        path,
        quantize=False,
        device=None
    ) -> Self:
        """
        Load the model from a pretrained model,
        initialize based on config.
        """

        device = device or torch.device("cuda")

        model = LlamaLora(
            tokenizer_path=os.path.join(path, "tokenizer"),
            quantize=quantize,
            device=device
        )

        assert os.path.exists(os.path.join(path, "lora")), "LoRA model does not exist."

        model._load_lora(os.path.join(path, "lora"))

        return model

    def save_pretrained(self, path):
        """
        Save the model to a specified path.
        """

        super().save_pretrained(path, save_modules=["lora"])

    def _prepare_llama_input(self, hints, descriptions):
        """Prepare input_ids, attentions."""

        hint_embeds = self.llama_embeddings(hints)
        description_embeds = self.llama_embeddings(descriptions)

        task_embed = self.task_embed

        # <|bos|> + <|task|> + hint + description
        inputs_embeds = torch.cat((
            self.bos_embed.unsqueeze(0).repeat(hint_embeds.size(0), 1, 1).detach(),
            task_embed.unsqueeze(0).repeat(hint_embeds.size(0), 1, 1).detach(),
            hint_embeds,
            description_embeds
        ), dim=1)

        attentions = torch.ones(inputs_embeds.size()[:-1], device=self.device)

        label_ids = torch.cat((
            torch.full(
                (torch.tensor(hint_embeds.size()[:-1]) + torch.tensor([0, 2])).tolist(),
                -100,
                device=self.device
            ),
            descriptions
        ), dim=1)

        return inputs_embeds, attentions, label_ids

    def forward(
        self,
        image_batches = None,
        labels = None,
    ):
        hints, descriptions = labels

        inputs_embeds, attentions, label_ids = self._prepare_llama_input(hints, descriptions)

        llama_out = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            labels=label_ids
        )

        return llama_out

    def generate(
        self,
        image_batches=None,
        hints=None,
        labels=None,
        do_decode=True,
        **generation_kwargs
    ):
        assert hints is not None or labels is not None, "Either hints or labels with hints must be provided."

        hints = hints if hints is not None else labels[0]

        # create empty labels of shape (bs, 0)
        labels = torch.empty(
            (hints.size(0), 0),
            dtype=torch.long,
            device=self.device
        )

        inputs_embeds, attentions, label_ids = self._prepare_llama_input(hints, labels)

        generated = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs
        )

        if do_decode:
            out = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return out

        return generated
