import os
import sys

import torch
import torchvision.datasets as dset

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data import *
from modeling import *
from testing import *


device = torch.device("cuda:3")

# ds = dset.CocoCaptions(
#     root = '/home/xbuban1/coco/images/val2017',
#     # root = '/mnt/gryf/home/xbuban1/coco/images/train2017',
#     annFile = '/home/xbuban1/coco/annotations/captions_val2017.json'
#     # annFile = '/mnt/gryf/home/xbuban1/coco/annotations/captions_train2017.json'
# )

llama_model = LlamaGameDescription.from_pretrained(
    "/home/xbuban1/LlamaGames/runs/8_Games/models/model_10",
    task="description",
    device=device
)

data_config = dict(
    root="/home/xbuban1/Games",
    data_name="apps_filtered.json",
    image_size=224,
    max_image_stack_size=10,
    max_label_length=1024,
    minibatch_size=1,
    data_split=0.8,
    seed=42
)

train_loader, val_loader = load_app_data(
    **data_config,
    tokenizer=llama_model.tokenizer,
    processor=llama_model.vit_processor,
    device=device
)

# ved_model = VEDModel(
#     "/home/xbuban1/LlamaGames/models/ved_model",
#     "nlpconnect/vit-gpt2-image-captioning",
#     device=device
# )

# caption_dataset(ds, ved_model, device, "captions/ved.json")
# caption_coco(ds, llama_model, device, "captions/finetune.json")
caption_app(val_loader, llama_model, "captions/games.json")
