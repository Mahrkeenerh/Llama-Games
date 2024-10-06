import json
import os
import sys

import torch

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data import *
from modeling import *
from training import *


def load_coco_model(
    image_merge_factor,
    vit_path,
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
    )

    model = LlamaCaption(
        projector_config=dict(
            image_merge_factor=image_merge_factor
        ),
        lora_config=lora_config,
        device=device
    )
    model.load_vit(vit_path)
    model.vit.eval()
    model.freeze(model.vit)

    if local_rank is not None:
        model.distribute(local_rank)

    return model


def load_game_model(
    model_path,
    quantize,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_bias,
    device,
    local_rank,
    **kwargs
):
    model = LlamaGameDescription.from_pretrained(
        model_path,
        quantize=quantize,
        load_modules=['vit', 'projector'],
        device=device
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=["embed_tokens", "q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model.init_lora(lora_config)
    model.freeze(model.vit, model.projector)

    # model.freeze(model.llama)
    for name, param in model.llama.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    if local_rank is not None:
        model.distribute(local_rank)

    return model


def load_lora_only_model(
    model_path,
    quantize,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_bias,
    device,
    local_rank,
    **kwargs
):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=["embed_tokens", "q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model = LlamaLora(
        quantize=quantize,
        lora_config=lora_config,
        device=device
    )

    if local_rank is not None:
        model.distribute(local_rank)

    return model


def load_coco_finetune_model(
    model_path,
    vit_path,
    device,
    local_rank,
    **kwargs
):
    model = LlamaCaption.from_pretrained(
        model_path,
        device=device
    )

    model.load_vit(vit_path)
    model.freeze(model.llama)

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
    run_name = f"runs/{run_i}_Games_quant"
    os.makedirs(run_name, exist_ok=True)

    model_config = dict(
        model_path="/home/xbuban1/LlamaGames/runs/6_3.1_Captions_224_merge/models/model_10",
        quantize=True,
        image_merge_factor=4,
        vit_path="models/vit_224",
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none"
    )

    # data_config = dict(
    #     root="/home/xbuban1/coco",
    #     image_size=224,
    #     max_label_length=64,
    #     minibatch_size=4,
    #     seed=42
    # )
    data_config = dict(
        root="/home/xbuban1/Games",
        data_name="apps_filtered_en.json",
        image_size=224,
        max_image_stack_size=10,
        max_label_length=1024,
        minibatch_size=1,
        data_split=0.8,
        include_hints=False,
        seed=42
    )

    train_config = dict(
        epochs=10,
        learning_rate=1e-5,
        grad_accum_steps=64 // data_config["minibatch_size"],
        num_samples=12,
        max_new_tokens=1024
    )

    save_config(
        run_name,
        model_config,
        data_config,
        train_config,
        local_rank
    )

    # model = load_coco_model(
    model = load_game_model(
    # model = load_lora_only_model(
    # model = load_coco_finetune_model(
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

    # train_loader, val_loader = load_coco_data(
    #     **data_config,
    #     tokenizer=model.tokenizer,
    #     processor=model.vit_processor,
    #     device=device,
    #     local_rank=local_rank
    # )
    train_loader, val_loader = load_app_data(
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
