import json
import os

from peft import LoraConfig
import torch

import modeling
import training


def load_vixtral(
    image_size,
    image_merge_factor,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_bias,
    device,
    local_rank
):
    vit_load = lambda : modeling.load_VED_vit(
        model_path="/home/xbuban1/ved_model",
        image_size=image_size,
        device=device
    )
    mixtral_load = lambda : modeling.load_mixtral(
        model_path="mistralai/Mixtral-8x7B-v0.1",
        load_4bit=True,
        device=device
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM"
    )

    vixtral = modeling.Vixtral(
        vit_load_func=vit_load,
        image_size=image_size,
        image_merge_factor=image_merge_factor,
        mixtral_load_func=mixtral_load,
        lora=lora_config,
        projector_path=None,
        device=device
    )

    if local_rank is not None:
        vixtral.distribute(local_rank)

    return vixtral


def save_config(
    run_name,
    vixtral_config,
    data_config,
    train_config,
    local_rank
):
    if not (local_rank is None or local_rank == 0):
        return

    with open(f"{run_name}/config.json", "w") as f:
        json.dump({
            "vixtral": vixtral_config,
            "data": data_config,
            "train": train_config
        }, f, indent=4)


def main():
    local_rank, run_i, device = training.init()
    run_name = f"runs/Vixtral_{run_i}"
    os.makedirs(run_name, exist_ok=True)

    vixtral_config = {
        "image_size": 448,
        "image_merge_factor": 4,
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_bias": "none"
    }

    data_config = {
        "root": "/home/xbuban1/Games",
        "data_name": "apps_filtered.json",
        "image_size": vixtral_config["image_size"],
        "max_image_stack_size": 6,
        "max_label_length": 512,
        "minibatch_size": 1,
        "data_split": 0.8,
        "seed": 42
    }

    train_config = {
        "epochs": 10,
        "learning_rate": 1e-4,
        "grad_accum_steps": 64,
        "num_samples": 60,
        "max_new_tokens": 512
    }

    save_config(
        run_name,
        vixtral_config,
        data_config,
        train_config,
        local_rank
    )

    vixtral = load_vixtral(
        **vixtral_config,
        device=device,
        local_rank=local_rank
    )
    vixtral.print_parameters()

    if local_rank is None or local_rank == 0:
        print(f"Training run {run_i}")
        print('-' * 100)

    optimizer = vixtral.set_optimizer(
        torch.optim.Adam,
        lr=train_config["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["epochs"])

    train_loader, val_loader = training.load_data(
        **data_config,
        tokenizer=vixtral.tokenizer,
        processor=vixtral.vit_processor,
        device=device,
        local_rank=local_rank
    )

    callbacks = [
        training.GenerateCallback(
            num_samples=train_config["num_samples"],
            max_new_tokens=train_config["max_new_tokens"],
            run_name=run_name,
            local_rank=local_rank
        ),
        training.SaveCallback(
            run_name=run_name,
            local_rank=local_rank
        ),
        training.LogCallback(
            run_name=run_name,
            local_rank=local_rank
        )
    ]

    training.train(
        vixtral=vixtral,
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
