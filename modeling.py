from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import LlamaTokenizerFast, MixtralForCausalLM, BitsAndBytesConfig, VisionEncoderDecoderModel, ViTImageProcessor 


def load_VED_vit(model_path, device):
    """Load the ViT model from a VED model."""

    ved_model = VisionEncoderDecoderModel.from_pretrained(
        model_path,
        device_map=device
    )

    # delete the language head
    del ved_model.decoder
    torch.cuda.empty_cache()

    vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_processor.size = {"width": 448, "height": 448}

    return ved_model.encoder, vit_processor


def load_mixtral(model_path, load_4bit, device):
    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_path,
        device_map=device
    )

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )

    mixtral = MixtralForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device
    )

    return mixtral, tokenizer


class Vixtral(torch.nn.Module):
    def __init__(
            self,
            image_size=448,
            vit_load_func=None,
            mixtral_load_func=None,
            lora_r=64,
            projector_path=None,
            device=None
        ):
        assert device is not None, "Device must be provided."
        super(Vixtral, self).__init__()

        self.device = device
        self.is_distributed = False

        if vit_load_func is None:
            self.vit = None
            self.vit_processor = None
        else:
            self.vit, self.vit_processor = vit_load_func()
            self._init_vit(image_size)

        assert mixtral_load_func is not None, "Mixtral model must be provided."

        self.mixtral, self.tokenizer = mixtral_load_func()
        self._init_mixtral(lora_r)

        self._init_embed_projector(projector_path)

    def _init_vit(self, image_size):
        """Apply custom modifications to the ViT model
        - Double input image size
        - Freeze the whole model
        """

        if self.vit is None:
            return

        self.vit.embeddings.patch_embeddings.image_size = [image_size, image_size]

        # Upscale position embeddings
        old_position_embeddings = self.vit.embeddings.position_embeddings
        first_row = old_position_embeddings[:, 0, :].clone().unsqueeze(0)
        rest_upscaled = torch.nn.functional.interpolate(
            old_position_embeddings[:, 1:, :].clone().unsqueeze(0),
            size=((image_size // 16) ** 2, self.vit.embeddings.patch_embeddings.projection.out_channels),
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

    def _init_mixtral(self, lora_r):
        """Apply custom modifications to the Mixtral model
        - Add LoRA
        - Freeze the rest of the model
        """

        if lora_r > 0:
            mixtral = prepare_model_for_kbit_training(self.mixtral)
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.mixtral = get_peft_model(mixtral, loraconfig)

        # Freeze the whole model
        else:
            for param in self.mixtral.parameters():
                param.requires_grad = False

        self.mixtral.eval()

    def _init_embed_projector(self, projector_path):
        """Initialize the projector from ViT to Mixtral embeds."""

        image_embed_size = self.vit.embeddings.patch_embeddings.projection.out_channels
        word_embed_size = self.mixtral.get_input_embeddings().embedding_dim

        self.embed_projector = torch.nn.Linear(
            image_embed_size * 4,
            word_embed_size,
            device=self.device
        )

        if projector_path is not None:
            self.load_projector(projector_path)

    def save_projector(self, path):
        if self.is_distributed:
            projector = self.embed_projector.module
        else:
            projector = self.embed_projector

        torch.save(projector.state_dict(), path)

    def load_projector(self, path):
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

        mixtral_params = sum(p.numel() for p in self.mixtral.parameters())
        mixtral_trainable_params = sum(p.numel() for p in self.mixtral.parameters() if p.requires_grad)
        print(f"Mixtral trainable params: {mixtral_trainable_params:,} || all params: {mixtral_params:,} || trainable/all: {mixtral_trainable_params / mixtral_params * 100:.2f}%")

        projector_params = sum(p.numel() for p in self.embed_projector.parameters())
        projector_trainable_params = sum(p.numel() for p in self.embed_projector.parameters() if p.requires_grad)
        print(f"Projector trainable params: {projector_trainable_params:,} || all params: {projector_params:,} || trainable/all: {projector_trainable_params / projector_params * 100:.2f}%")

    def _prepare_mixtral_input(self, project_embed, labels):
        """Concat project_embeds with labels, prepare input_ids, attentions."""

        label_embeds = self.mixtral.get_input_embeddings()(labels)
        inputs_embeds = torch.cat((project_embed, label_embeds), dim=1)

        attentions = torch.ones(inputs_embeds.size()[:-1], device=self.device)

        label_ids = torch.cat((torch.full((project_embed.size()[:-1]), -100, device=self.device), labels), dim=1)

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
            enc_outs = []
            for image_batch in image_batches:
                image_embeds = self.vit(image_batch).last_hidden_state
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                enc_outs.append(image_embeds)

            encoder_outputs = torch.cat(enc_outs, dim=1)

        projected_embed = self.embed_projector(encoder_outputs).type(torch.float16)

        inputs_embeds, attentions, label_ids = self._prepare_mixtral_input(projected_embed, labels)

        mixtral_out = self.mixtral(
            inputs_embeds=inputs_embeds,
            attention_mask=attentions,
            labels=label_ids
        )

        return mixtral_out
