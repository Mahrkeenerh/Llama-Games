import json
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import app_dataset


class Callback:
    def __init__(self, only_main=True, **kwargs):
        self.only_main = only_main
        self.run_name = kwargs.get("run_name", None)
        self.local_rank = kwargs.get("local_rank", None)
        self.is_main = self.local_rank is None or self.local_rank == 0

    def start(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass

    def train_step(self, **kwargs):
        pass

    def train_epoch(self, **kwargs):
        pass

    def val_epoch(self, **kwargs):
        pass


class Callbacks:
    def __init__(self, callbacks, **kwargs):
        self.callbacks = callbacks

    def start(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.start(**kwargs)

    def end(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.end(**kwargs)

    def train_step(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.train_step(**kwargs)

    def train_epoch(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.train_epoch(**kwargs)

    def val_epoch(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.val_epoch(**kwargs)


class GenerateCallback(Callback):
    def __init__(self, num_samples, max_new_tokens, **kwargs):
        super().__init__(only_main=False, **kwargs)

        self.target_dir = "generated"
        if self.run_name is not None:
            self.target_dir = os.path.join(self.run_name, self.target_dir)

        if self.is_main:
            os.makedirs(os.path.join(self.target_dir, "temp"), exist_ok=True)

        divider = torch.distributed.get_world_size() if self.local_rank is not None else 1

        self.local_num_samples = num_samples // divider
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def sample_generate(
        self,
        vixtral,
        epoch,
        data_loader
    ):
        temp_dir = os.path.join(self.target_dir, "temp")
        data_iter = iter(data_loader)

        generates = []
        bar = tqdm(
            range(self.local_num_samples),
            desc="Generating text descriptions"
        ) if self.is_main is None or self.is_main == 0 else range(self.local_num_samples)
        for _ in bar:
            image_batches, labels = next(data_iter)

            generated = vixtral.generate(
                image_batches=image_batches,
                max_new_tokens=self.max_new_tokens
            )
            generates.append(generated)

        with open(
            os.path.join(temp_dir, f"{self.local_rank}.json"),
            "w"
        ) as f:
            json.dump(generates, f, indent=4)

        if self.local_rank is not None:
            torch.distributed.barrier()

        if self.is_main:
            all_generates = []
            for file_name in os.listdir(temp_dir):
                with open(os.path.join(temp_dir, file_name), "r") as f:
                    all_generates.extend(json.load(f))

            with open(
                os.path.join(self.target_dir, f"sample_{epoch}.json"),
                "w"
            ) as f:
                json.dump(all_generates, f, indent=4)

            for file_name in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file_name))

    def start(self, **kwargs):
        self.sample_generate(
            vixtral=kwargs["model"],
            epoch=0,
            data_loader=kwargs["val_loader"]
        )

    def train_epoch(self, **kwargs):
        self.sample_generate(
            vixtral=kwargs["model"],
            epoch=kwargs["epoch"] + 1,
            data_loader=kwargs["val_loader"]
        )


class SaveCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.is_main:
            return

        self.target_dir = "models"
        if self.run_name is not None:
            self.target_dir = os.path.join(self.run_name, self.target_dir)

        os.makedirs(self.target_dir, exist_ok=True)

    def start(self, **kwargs):
        kwargs["model"].save_pretrained(
            os.path.join(self.target_dir, "vixtral_0")
        )

    def train_epoch(self, **kwargs):
        kwargs["model"].save_pretrained(
            os.path.join(self.target_dir, f"vixtral_{kwargs['epoch'] + 1}")
        )


class LogCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target_dir = "losses"
        if self.run_name is not None:
            self.target_dir = os.path.join(self.run_name, self.target_dir)

        os.makedirs(self.target_dir, exist_ok=True)

        self.train_losses = [[]]
        self.rolling_losses = [[]]
        self.val_losses = []
        self.lrs = []

    def train_step(self, **kwargs):
        self.train_losses[-1].append(kwargs["loss"].item())
        self.rolling_losses[-1].append(kwargs["rolling_loss"])

    def val_epoch(self, **kwargs):
        self.val_losses.append(kwargs["loss"])
        self.lrs.append(kwargs["lr"])

        with open(
            os.path.join(self.target_dir, "loss.json"),
            "w"
        ) as f:
            json.dump(
                {
                    "train": self.train_losses,
                    "rolling": self.rolling_losses,
                    "val": self.val_losses,
                    "lrs": self.lrs
                },
                f,
                indent=4
            )

        plt.figure(figsize=(10, 5))
        plt.plot([loss for loss in self.train_losses[-1]], label="Train loss")
        plt.plot([loss for loss in self.rolling_losses[-1]], label="Rolling loss")
        plt.plot([loss for loss in [self.val_losses[-1]] * len(self.train_losses[-1])], label="Validation loss")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Losses - Epoch {kwargs['epoch'] + 1}")
        plt.savefig(os.path.join(self.target_dir, f"loss_{kwargs['epoch'] + 1}.png"))

        self.train_losses.append([])
        self.rolling_losses.append([])

    def end(self, **kwargs):
        del self.train_losses[-1]
        del self.rolling_losses[-1]

        with open(
            os.path.join(self.target_dir, "loss.json"),
            "w"
        ) as f:
            json.dump(
                {
                    "train": self.train_losses,
                    "rolling": self.rolling_losses,
                    "val": self.val_losses,
                    "lrs": self.lrs
                },
                f,
                indent=4
            )

        # Create plots
        plt.figure(figsize=(10, 5))
        plt.plot([loss for epoch_losses in self.train_losses for loss in epoch_losses], label="Train loss")
        plt.plot([loss for epoch_losses in self.rolling_losses for loss in epoch_losses], label="Rolling loss")
        plt.plot([loss for epoch_losses in self.val_losses for loss in [epoch_losses] * len(self.train_losses[0])], label="Validation loss")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training loss, rolling loss, and validation loss")
        plt.savefig(os.path.join(self.target_dir, "loss.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation loss")
        plt.savefig(os.path.join(self.target_dir, "val_loss.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(self.lrs, label="Learning rate")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning rate")
        plt.savefig(os.path.join(self.target_dir, "lr.png"))


def init():
    is_distributed = "LOCAL_RANK" in os.environ

    run_i = len(os.listdir("runs")) if os.path.exists("runs") else 0

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier()
    else:
        local_rank = None
        device = torch.device("cuda")


    return local_rank, run_i, device


def load_data(
    root,
    data_name,
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
    dataset_tokenizer = tokenizer if minibatch_size == 1 else None
    dataset = app_dataset.AppDataset(
        root=root,
        data_name=data_name,
        images_subdir="images",
        max_image_stack_size=max_image_stack_size,
        tokenizer=dataset_tokenizer,
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

    dataloader_tokenizer = tokenizer if minibatch_size != 1 else None
    train_loader = app_dataset.AppDataLoader(
        train_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        tokenizer=dataloader_tokenizer,
        image_size=image_size,
        device=device,
        sampler=train_sampler
    )

    val_loader = app_dataset.AppDataLoader(
        val_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        tokenizer=dataloader_tokenizer,
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

    callbacks.start(model=vixtral, val_loader=val_loader)

    for epoch in range(epochs):
        optimizer.zero_grad()
        grad_accum_step = 0

        train_bar = get_bar_iter(True, train_loader, epoch, epochs, local_rank)
        for data in train_bar:
            image_batches, labels = data

            loss = vixtral(image_batches, labels).loss.mean() / grad_accum_steps
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

        callbacks.train_epoch(
            model=vixtral,
            epoch=epoch,
            val_loader=val_loader
        )

        # Validation
        with torch.no_grad():
            val_loss = 0
            val_bar = get_bar_iter(False, val_loader, epoch, epochs, local_rank)
            for i, data in enumerate(val_bar):
                image_batches, labels = data

                step_loss = vixtral(image_batches, labels).loss.mean()
                torch.distributed.reduce(step_loss, 0, op=torch.distributed.ReduceOp.AVG)
                val_loss += step_loss.item()

                set_bar_description(False, val_bar, epoch, epochs, val_loss / (i + 1), local_rank=local_rank)

            callbacks.val_epoch(
                epoch=epoch,
                loss=val_loss / len(val_loader),
                lr=scheduler.get_last_lr()[0]
            )

    callbacks.end()
