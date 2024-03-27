import os

from peft import LoraConfig
import torch
from tqdm import tqdm

import app_dataset
import modeling


def init():
    is_distributed = "LOCAL_RANK" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        local_rank = 0
        device = torch.device("cuda")

    return local_rank, device


def load_vixtral(
    image_size,
    lora_r,
    lora_alpha,
    lora_dropout,
    device,
    local_rank
):
    vit_load = lambda : modeling.load_VED_vit(
        model_path="/home/xbuban1/ved_model",
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
        bias="none",
        task_type="CAUSAL_LM"
    )

    vixtral = modeling.Vixtral(
        vit_load_func=vit_load,
        image_size=image_size,
        mixtral_load_func=mixtral_load,
        lora=lora_config,
        projector_path=None,
        device=device
    )

    if local_rank is not None:
        vixtral.distribute(local_rank)

    return vixtral


def load_data(
    tokenizer,
    processor,
    image_size,
    max_image_stack_size,
    max_label_length,
    minibatch_size,
    data_split,
    device,
    seed,
    local_rank
):
    dataset = app_dataset.AppDataset(
        root="/home/xbuban1/Games",
        data_name="apps_filtered.json",
        images_subdir="images",
        max_image_stack_size=max_image_stack_size,
        # set tokenizer in dataloader if minibatch size > 1
        tokenizer=tokenizer,
        max_label_length=max_label_length,
        device=device,
        seed=seed
    )

    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [data_split, 1 - data_split],
        generator=torch.Generator().manual_seed(seed)
    )

    if local_rank is not None:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = app_dataset.AppDataLoader(
        train_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        image_size=image_size,
        device=device,
        sampler=train_sampler
    )

    val_loader = app_dataset.AppDataLoader(
        val_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        image_size=image_size,
        device=device,
        sampler=val_sampler
    )

    return train_loader, val_loader


def train(
    vixtral,
    optimizer,
    scheduler,
    train_loader, 
    val_loader,
    epochs,
    grad_accum_steps,
    local_rank
):

    def get_bar_iter(is_train, loader, epoch, epochs, local_rank):
        text = "Training" if is_train else "Validation"
        rolling_text = f"Rolling loss: - | " if is_train else ""
        if local_rank is None or local_rank == 0:
            bar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{epochs} - {text}, {rolling_text}Loss: -"
            )
        else:
            bar = loader

        return bar

    def set_bar_description(
        is_train,
        bar,
        epoch,
        epochs,
        loss,
        rolling_loss=None,
        local_rank=None
    ):
        text = "Training" if is_train else "Validation"
        roll_text = f"Rolling loss: {rolling_loss:.4f} | " if rolling_loss is not None else ""

        if local_rank is None or local_rank == 0:
            bar.set_description(
                f"Epoch {epoch + 1}/{epochs} - {text}, {roll_text}Loss: {loss:.4f}"
            )

    def get_rolling_loss(rolling_loss, loss):
        if rolling_loss == 0:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

        return rolling_loss

    grad_accum_steps = grad_accum_steps / torch.distributed.get_world_size() if local_rank is not None else grad_accum_steps
    rolling_loss = 0

    vixtral.save_pretrained("vixtral_0", local_rank=local_rank)

    for epoch in range(epochs):
        optimizer.zero_grad()
        grad_accum_step = 0

        train_bar = get_bar_iter(True, train_loader, epoch, epochs, local_rank)
        for data in train_bar:
            image_batches, labels = data

            loss = vixtral(image_batches, labels).loss.mean() / grad_accum_steps
            rolling_loss = get_rolling_loss(rolling_loss, loss * grad_accum_steps)

            set_bar_description(True, train_bar, epoch, epochs, loss * grad_accum_steps, rolling_loss, local_rank)

            loss.backward()

            grad_accum_step += 1
            if grad_accum_step == grad_accum_steps:
                grad_accum_step = 0
                optimizer.step()
                optimizer.zero_grad()

        # Gradient accumulation for the last batch
        if grad_accum_step > 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        vixtral.save_pretrained(f"vixtral_{epoch + 1}", local_rank=local_rank)

        with torch.no_grad():
            val_loss = 0
            val_bar = get_bar_iter(False, val_loader, epoch, epochs, local_rank)
            for i, data in enumerate(val_bar):
                image_batches, labels = data

                step_loss = vixtral(image_batches, labels).loss.mean()
                torch.distributed.reduce(step_loss, 0, op=torch.distributed.ReduceOp.AVG)
                val_loss += step_loss.item()

                set_bar_description(False, val_bar, epoch, epochs, val_loss / (i + 1), local_rank=local_rank)


def main():
    local_rank, device = init()

    image_size = 448

    vixtral = load_vixtral(
        image_size,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        device=device,
        local_rank=local_rank
    )
    vixtral.print_parameters()

    epochs = 10

    optimizer = vixtral.set_optimizer(torch.optim.Adam, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader = load_data(
        tokenizer=vixtral.tokenizer,
        processor=vixtral.vit_processor,
        image_size=image_size,
        max_image_stack_size=6,
        max_label_length=512,
        minibatch_size=1,
        data_split=0.8,
        device=device,
        seed=42,
        local_rank=local_rank
    )

    train(
        vixtral,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        epochs=epochs,
        grad_accum_steps=64,
        local_rank=local_rank
    )


if __name__ == "__main__":
    main()
