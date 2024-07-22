from tqdm import tqdm

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from datasets import *
from LlamaGames.modeling.llama_games import *


def preprocess(t):
    device = torch.device("cuda")

    vit = ViTModel.from_pretrained("/home/xbuban1/LlamaGames/models/vit", device_map=device)
    vit_processor = ViTImageProcessor.from_pretrained("/home/xbuban1/LlamaGames/models/vit")

    ds = dset.CocoCaptions(
        root=f"/home/xbuban1/coco/images/{t}2017",
        annFile=f"/home/xbuban1/coco/annotations/captions_{t}2017.json",
        transform=transforms.PILToTensor()
    )

    embeds_out = []
    # cls_out = []
    for image, ann in tqdm(ds):
        image = vit_processor([image], return_tensors="pt").pixel_values.to(device)
        enc_out = vit(image).last_hidden_state
        # enc_out_cls = enc_out[:, 0, :].unsqueeze(1)

        embeds_out.append(enc_out.to("cpu"))
        # cls_out.append(enc_out_cls.to("cpu"))

    torch.save(embeds_out, f"/home/xbuban1/coco/{t}_embeds.pt")


@torch.no_grad()
def main():
    preprocess("val")


if __name__ == "__main__":
    main()
