import os

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


def load_vixtral(image_size, lora_r, device, local_rank):
    vit_load = lambda : modeling.load_VED_vit(
        model_path="/home/xbuban1/ved_model",
        device=device
    )
    mixtral_load = lambda : modeling.load_mixtral(
        model_path="mistralai/Mixtral-8x7B-v0.1",
        load_4bit=True,
        device=device
    )

    vixtral = modeling.Vixtral(
        vit_load_func=vit_load,
        image_size=image_size,
        mixtral_load_func=mixtral_load,
        lora_r=lora_r,
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

    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))

    sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if local_rank is not None else None

    train_loader = app_dataset.AppDataLoader(
        train_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        image_size=image_size,
        device=device,
        sampler=sampler
    )

    val_loader = app_dataset.AppDataLoader(
        val_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        image_size=image_size,
        device=device,
        sampler=sampler
    )

    return train_loader, val_loader


def train(vixtral, optimizer, train_loader, val_loader, epochs, grad_accum_steps, local_rank):

    def get_train_bar(train_loader, epoch, epochs, local_rank):
        if local_rank is None or local_rank == 0:
            train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} - Training, Loss: -"
            )
        else:
            train_bar = train_loader

        return train_bar

    def set_train_bar_description(train_bar, epoch, epochs, loss, local_rank):
        if local_rank is None or local_rank == 0:
            train_bar.set_description(
                f"Epoch {epoch + 1}/{epochs} - Training, Loss: {loss}"
            )

    for epoch in range(epochs):
        optimizer.zero_grad()
        grad_accum_step = 0

        train_bar = get_train_bar(train_loader, epoch, epochs, local_rank)
        for data in train_bar:
            image_batches, labels = data

            loss = vixtral(image_batches, labels).loss.mean() / grad_accum_steps

            set_train_bar_description(train_bar, epoch, epochs, loss * grad_accum_steps, local_rank)

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

        # TODO
        # Do evaluation
        # Save model


def main():
    local_rank, device = init()

    image_size = 448

    vixtral = load_vixtral(
        image_size,
        lora_r=64,
        device=device,
        local_rank=local_rank
    )
    vixtral.print_parameters()

    optimizer = vixtral.set_optimizer(torch.optim.Adam, lr=1e-4)

    train_loader, val_loader = load_data(
        tokenizer=vixtral.tokenizer,
        processor=vixtral.vit_processor,
        image_size=image_size,
        max_image_stack_size=6,
        max_label_length=512,
        minibatch_size=1,
        device=device,
        seed=42,
        local_rank=local_rank
    )

    train(
        vixtral,
        optimizer,
        train_loader,
        val_loader,
        epochs=10,
        grad_accum_steps=64,
        local_rank=local_rank
    )


if __name__ == "__main__":
    main()
