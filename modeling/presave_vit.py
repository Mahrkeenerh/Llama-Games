import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor


def presave_vit(model_id, image_size, out):
    """Load the ViT model from the VED model."""

    # Load VED model
    ved_model = VisionEncoderDecoderModel.from_pretrained(model_id)
    vit = ved_model.encoder

    # delete the language head
    del ved_model.decoder

    vit_processor = ViTImageProcessor.from_pretrained(model_id)
    vit_processor.size = {"width": image_size, "height": image_size}

    vit.embeddings.patch_embeddings.image_size = [image_size, image_size]
    vit.config.image_size = image_size

    # Upscale position embeddings
    if image_size != 224:
        old_position_embeddings = vit.embeddings.position_embeddings
        first_row = old_position_embeddings[:, 0, :].clone().unsqueeze(0)
        rest_upscaled = torch.nn.functional.interpolate(
            old_position_embeddings[:, 1:, :].clone().unsqueeze(0),
            size=((image_size // 16) ** 2, vit.embeddings.patch_embeddings.projection.out_channels),
            mode="nearest"
        )[0]

        new_position_embeddings = torch.nn.Parameter(
            torch.cat((first_row, rest_upscaled), dim=1),
            requires_grad=True
        )

        vit.embeddings.position_embeddings = new_position_embeddings

    # Disable unused layer
    vit.pooler = None

    vit.save_pretrained(out)
    vit_processor.save_pretrained(out)


if __name__ == "__main__":
    # Modification of the nlpconnect/vit-gpt2-image-captioning
    # with output_hidden_states enabled
    presave_vit("/home/xbuban1/LlamaGames/models/ved_model", 224, "/home/xbuban1/LlamaGames/models/vit_smol")
