import os
import sys

import torch
import torchvision.datasets as dset

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from modeling import *
from testing import *


device = torch.device("cuda:2")

ds = dset.CocoCaptions(
    root = '/home/xbuban1/coco/images/val2017',
    # root = '/mnt/gryf/home/xbuban1/coco/images/train2017',
    annFile = '/home/xbuban1/coco/annotations/captions_val2017.json'
    # annFile = '/mnt/gryf/home/xbuban1/coco/annotations/captions_train2017.json'
)

llama_model = LlamaGameDescription.from_pretrained(
    "/home/xbuban1/LlamaGames/runs/3_Llama_Captions_448_no_merge/models/model_10",
    task="caption",
    device=device
)

# ved_model = VEDModel(
#     "/home/xbuban1/LlamaGames/models/ved_model",
#     "nlpconnect/vit-gpt2-image-captioning",
#     device=device
# )

# caption_dataset(ds, ved_model, device, "captions/ved.json")
caption_dataset(ds, llama_model, device, "captions/val_448_no_merge.json")
