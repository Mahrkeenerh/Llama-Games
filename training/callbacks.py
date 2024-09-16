import json
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


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

    def before_train_epoch(self, **kwargs):
        pass

    def after_train_epoch(self, **kwargs):
        pass

    def after_val_epoch(self, **kwargs):
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

    def before_train_epoch(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.before_train_epoch(**kwargs)

    def after_train_epoch(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.after_train_epoch(**kwargs)

    def after_val_epoch(self, **kwargs):
        for callback in self.callbacks:
            if not callback.only_main or callback.is_main:
                callback.after_val_epoch(**kwargs)


class GenerateCallback(Callback):
    def __init__(self, tokenizer, num_samples, max_new_tokens, **kwargs):
        super().__init__(only_main=False, **kwargs)

        self.target_dir = "generated"
        if self.run_name is not None:
            self.target_dir = os.path.join(self.run_name, self.target_dir)

        if self.is_main:
            os.makedirs(os.path.join(self.target_dir, "temp"), exist_ok=True)

        divider = torch.distributed.get_world_size() if self.local_rank is not None else 1

        self.tokenizer = tokenizer
        self.local_num_samples = num_samples // divider
        self.max_new_tokens = max_new_tokens

    def save_targets(self, data_loader, epoch, sub):
        temp_dir = os.path.join(self.target_dir, "temp")
        data_iter = iter(data_loader)

        targets = []
        for _ in range(self.local_num_samples):
            image_batches, labels = next(data_iter)
            labels = [
                self.tokenizer.decode(label, skip_special_tokens=True) 
                for label in labels
            ]
            targets.extend(labels)

        with open(
            os.path.join(temp_dir, f"{self.local_rank}.json"),
            "w",
            encoding="utf-16"
        ) as f:
            json.dump(targets, f, indent=4)

        if self.local_rank is not None:
            torch.distributed.barrier()

        if self.is_main:
            all_targets = []
            for file_name in os.listdir(temp_dir):
                with open(
                    os.path.join(temp_dir, file_name),
                    "r",
                    encoding="utf-16"
                ) as f:
                    all_targets.extend(json.load(f))

            with open(
                os.path.join(self.target_dir, f"targets_{epoch}_{sub}.json"),
                "w",
                encoding="utf-16"
            ) as f:
                json.dump(all_targets, f, indent=4)

            for file_name in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file_name))

    @torch.no_grad()
    def sample_generate(
        self,
        model,
        epoch,
        data_loader,
        sub
    ):
        temp_dir = os.path.join(self.target_dir, "temp")
        data_iter = iter(data_loader)

        generates = []
        bar = tqdm(
            range(self.local_num_samples),
            desc=f"Generating {sub} text descriptions"
        ) if self.is_main else range(self.local_num_samples)
        for _ in bar:
            image_batches, labels = next(data_iter)

            generated = model.generate(
                image_batches=image_batches,
                max_new_tokens=self.max_new_tokens
            )
            generates.append(generated)

        with open(
            os.path.join(temp_dir, f"{self.local_rank}.json"),
            "w",
            encoding="utf-16"
        ) as f:
            json.dump(generates, f, indent=4)

        if self.local_rank is not None:
            torch.distributed.barrier()

        if self.is_main:
            all_generates = []
            for file_name in os.listdir(temp_dir):
                with open(
                    os.path.join(temp_dir, file_name),
                    "r",
                    encoding="utf-16"
                ) as f:
                    all_generates.extend(json.load(f))

            with open(
                os.path.join(self.target_dir, f"sample_{epoch}_{sub}.json"),
                "w",
                encoding="utf-16"
            ) as f:
                json.dump(all_generates, f, indent=4)

            for file_name in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file_name))

    def start(self, **kwargs):
        self.after_train_epoch(**kwargs)
        self.after_val_epoch(**kwargs)

    def after_train_epoch(self, **kwargs):
        e = kwargs.get("epoch", -1) + 1

        self.save_targets(
            data_loader=kwargs["train_loader"],
            epoch=e,
            sub="train"
        )

        self.sample_generate(
            model=kwargs["model"],
            epoch=e,
            data_loader=kwargs["train_loader"],
            sub="train"
        )

    def after_val_epoch(self, **kwargs):
        e = kwargs.get("epoch", -1) + 1

        self.save_targets(
            data_loader=kwargs["val_loader"],
            epoch=e,
            sub="val"
        )

        self.sample_generate(
            model=kwargs["model"],
            epoch=e,
            data_loader=kwargs["val_loader"],
            sub="val"
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
            os.path.join(self.target_dir, "model_0")
        )

    def after_train_epoch(self, **kwargs):
        kwargs["model"].save_pretrained(
            os.path.join(self.target_dir, f"model_{kwargs['epoch'] + 1}")
        )


class LogCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target_dir = "losses"
        if self.run_name is not None:
            self.target_dir = os.path.join(self.run_name, self.target_dir)

        os.makedirs(self.target_dir, exist_ok=True)

        self.train_losses = []
        self.rolling_losses = []
        self.val_losses = []
        self.lrs = []

    def train_step(self, **kwargs):
        self.train_losses[-1].append(kwargs["loss"].item())
        self.rolling_losses[-1].append(kwargs["rolling_loss"])

    def before_train_epoch(self, **kwargs):
        self.train_losses.append([])
        self.rolling_losses.append([])
        self.lrs.append(kwargs["lr"])

    def after_val_epoch(self, **kwargs):
        self.val_losses.append(kwargs["loss"])

        plt.figure(figsize=(10, 5))
        plt.plot([loss for loss in self.train_losses[-1]], label="Train loss")
        plt.plot([loss for loss in self.rolling_losses[-1]], label="Rolling loss")
        plt.plot([loss for loss in [self.val_losses[-1]] * len(self.train_losses[-1])], label="Validation loss")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Losses - Epoch {kwargs['epoch'] + 1}")
        plt.savefig(os.path.join(self.target_dir, f"loss_{kwargs['epoch'] + 1}.png"))
        plt.close()

        self.end()

    def end(self, **kwargs):
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
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation loss")
        plt.savefig(os.path.join(self.target_dir, "val_loss.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.lrs, label="Learning rate")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning rate")
        plt.savefig(os.path.join(self.target_dir, "lr.png"))
        plt.close()
