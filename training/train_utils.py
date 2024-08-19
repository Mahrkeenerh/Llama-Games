import os

import torch
from tqdm import tqdm

from .callbacks import Callbacks


def init():
    is_distributed = "LOCAL_RANK" in os.environ

    run_is = []
    if os.path.exists("runs"):
        run_is = [int(run.split("_")[0]) for run in os.listdir("runs")]

    run_i = 0 if len(run_is) == 0 else sorted(run_is)[-1] + 1

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier()
    else:
        local_rank = None
        device = torch.device("cuda")

    return local_rank, run_i, device


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    grad_accum_steps,
    callbacks,
    local_rank
):

    def get_bar_iter(is_train, loader, epoch, epochs, local_rank):
        text = "Training" if is_train else "Validation"
        if local_rank is None or local_rank == 0:
            bar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{epochs} - {text}, Loss: -"
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
        local_rank=None
    ):
        text = "Training" if is_train else "Validation"

        if local_rank is None or local_rank == 0:
            bar.set_description(
                f"Epoch {epoch + 1}/{epochs} - {text}, Loss: {loss:.4f}"
            )

    def get_rolling_loss(rolling_loss, loss):
        if rolling_loss == 0:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

        return rolling_loss

    grad_accum_steps = grad_accum_steps / torch.distributed.get_world_size() if local_rank is not None else grad_accum_steps
    rolling_loss = 0

    if not isinstance(callbacks, Callbacks):
        callbacks = Callbacks(callbacks)

    callbacks.start(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        grad_accum_step = 0

        callbacks.before_train_epoch(
            lr=scheduler.get_last_lr()[0]
        )

        train_bar = get_bar_iter(True, train_loader, epoch, epochs, local_rank)
        for data in train_bar:
            image_batches, labels = data

            loss = model(image_batches, labels).loss.mean() / grad_accum_steps
            rolling_loss = get_rolling_loss(rolling_loss, loss * grad_accum_steps)

            set_bar_description(True, train_bar, epoch, epochs, rolling_loss, local_rank)

            loss.backward()

            grad_accum_step += 1
            if grad_accum_step == grad_accum_steps:
                grad_accum_step = 0
                optimizer.step()
                optimizer.zero_grad()

            callbacks.train_step(loss=loss * grad_accum_steps, rolling_loss=rolling_loss)

        # Gradient accumulation for the last batch
        if grad_accum_step > 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        callbacks.after_train_epoch(
            model=model,
            epoch=epoch,
            train_loader=train_loader
        )

        # Validation
        with torch.no_grad():
            val_loss = 0
            val_bar = get_bar_iter(False, val_loader, epoch, epochs, local_rank)
            for i, data in enumerate(val_bar):
                image_batches, labels = data

                step_loss = model(image_batches, labels).loss.mean()
                torch.distributed.reduce(step_loss, 0, op=torch.distributed.ReduceOp.AVG)
                val_loss += step_loss.item()

                set_bar_description(False, val_bar, epoch, epochs, val_loss / (i + 1), local_rank=local_rank)

            callbacks.after_val_epoch(
                model=model,
                epoch=epoch,
                val_loader=val_loader,
                loss=val_loss / len(val_loader)
            )

    callbacks.end()
