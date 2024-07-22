import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer


class VEDModel(torch.nn.Module):
    def __init__(
        self,
        model_id,
        tokenizer_id,
        device
    ):
        super(VEDModel, self).__init__()

        self.ved_model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_id)
        self.vit_processor = ViTImageProcessor.from_pretrained(model_id)
        self.device = device

    def generate(
        self,
        image,
        max_new_tokens=128
    ):
        image = self.vit_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.ved_model.generate(image.pixel_values, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
