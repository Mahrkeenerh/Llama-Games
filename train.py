import json
import os

import torch

from datasets import *
from modeling import *
from training import *


def load_model(
    task,
    image_merge_factor,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_bias,
    device,
    local_rank
):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=["embed_tokens"],
        task_type="CAUSAL_LM",
        # modules_to_save=["embed_tokens"]
    )

    model = LlamaGameDescription(
        task=task,
        projector_config=dict(
            image_merge_factor=image_merge_factor
        ),
        lora_config=lora_config,
        device=device
    )
    model.load_vit("models/vit")
    model.vit.eval()
    model.freeze(model.vit)

    if local_rank is not None:
        model.distribute(local_rank)

    return model


def save_config(
    run_name,
    model_config,
    data_config,
    train_config,
    local_rank
):
    if not (local_rank is None or local_rank == 0):
        return

    with open(f"{run_name}/config.json", "w") as f:
        json.dump({
            "model": model_config,
            "data": data_config,
            "train": train_config
        }, f, indent=4)


def main():
    local_rank, run_i, device = init()
    run_name = f"runs/{run_i}_Llama_Game_Desc"
    os.makedirs(run_name, exist_ok=True)

    # model_config = dict(
    #     image_merge_factor=4,
    #     lora_r=64,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     lora_bias="none"
    # )
    model_config = dict(
        task="caption",
        image_merge_factor=4,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none"
    )

    # data_config = {
    #     "root": "/home/xbuban1/Games",
    #     "data_name": "apps_filtered.json",
    #     "image_size": model_config["image_size"],
    #     "max_image_stack_size": 10,
    #     "max_label_length": 1024,
    #     "minibatch_size": 1,
    #     "data_split": 0.8,
    #     "seed": 42
    # }
    data_config = dict(
        train_root="/home/xbuban1/coco/images/train2017",
        train_ann_file="/home/xbuban1/coco/annotations/captions_train2017.json",
        val_root="/home/xbuban1/coco/images/val2017",
        val_ann_file="/home/xbuban1/coco/annotations/captions_val2017.json",
        image_size=448,
        max_label_length=1024,
        minibatch_size=16,
        seed=42
    )

    train_config = dict(
        epochs=10,
        learning_rate=1e-4,
        grad_accum_steps=64 / data_config["minibatch_size"],
        num_samples=1,
        max_new_tokens=1024
    )

    save_config(
        run_name,
        model_config,
        data_config,
        train_config,
        local_rank
    )

    model = load_model(
        **model_config,
        device=device,
        local_rank=local_rank
    )
    model.print_parameters()

    if local_rank is None or local_rank == 0:
        print(f"Training run {run_i}")
        print('-' * 100)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["epochs"])

    train_loader, val_loader = load_coco_data(
        **data_config,
        tokenizer=model.tokenizer,
        processor=model.vit_processor,
        device=device,
        local_rank=local_rank
    )

    callbacks = [
        GenerateCallback(
            tokenizer=model.tokenizer,
            num_samples=train_config["num_samples"],
            max_new_tokens=train_config["max_new_tokens"],
            run_name=run_name,
            local_rank=local_rank
        ),
        SaveCallback(
            run_name=run_name,
            local_rank=local_rank
        ),
        LogCallback(
            run_name=run_name,
            local_rank=local_rank
        )
    ]

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config["epochs"],
        grad_accum_steps=train_config["grad_accum_steps"],
        callbacks=callbacks,
        local_rank=local_rank
    )


if __name__ == "__main__":
    main()
