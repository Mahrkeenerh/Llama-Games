import json
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms


class DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer=None,
        max_label_length=512,
        device=None,
        seed=None
    ):
        self.tokenizer = tokenizer
        self.max_label_length = max_label_length
        self.device = device

        seed = seed or random.randint(0, 2 ** 32 - 1)
        self.rnd_state = np.random.RandomState(seed)

    def _tokenize(self, text):
        if self.tokenizer is None:
            return text

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt"
        ).input_ids[0]

        # Append <|end_of_text|> token if enough space
        if tokenized.shape[0] < self.max_label_length:
            tokenized = torch.cat(
                (tokenized, torch.tensor([self.tokenizer.eos_token_id]))
            )

        return tokenized


class AppDataset(DatasetBase):
    def __init__(
        self,
        root,
        data_name,
        images_subdir,
        max_image_stack_size=8,
        tokenizer=None,
        max_label_length=512,
        include_hints=False,
        device=None,
        seed=None
    ):
        self.max_image_stack_size = max_image_stack_size
        self.include_hints = include_hints

        super().__init__(
            tokenizer=tokenizer,
            max_label_length=max_label_length,
            device=device,
            seed=seed
        )

        self._load_datafile(root, data_name, images_subdir)

    def _load_datafile(self, root, data_name, images_subdir):
        data_json = json.load(open(os.path.join(root, data_name), "r", encoding="utf-8"))
        dataset_raw = {
            app["appId"]: {
                "image_paths": [],
                "title": app["title"],
                "summary": app["summary"],
                "description": app["description"]
            }
            for app in data_json
        }

        # Add image paths
        for image in os.listdir(os.path.join(root, images_subdir)):
            app_id = "_".join(image.split("_")[:-1])
            if app_id in dataset_raw:
                dataset_raw[app_id]["image_paths"].append(os.path.join(root, images_subdir, image))

        self.image_paths = []
        self.hints = []
        self.descriptions = []

        # Extract data into separate lists
        for app_id, app in dataset_raw.items():
            if len(app["image_paths"]) > 0:
                self.image_paths.append(app["image_paths"])
                if self.include_hints:
                    self.hints.append(f"Game title: {app['title']}\nShort summary: {app['summary']}\nLong description: ")
                else:
                    self.hints.append("")
                self.descriptions.append(app["description"])
            else:
                print("No images for", app_id)

        self.hint_labels = [None] * len(self.hints)
        self.description_labels =  [None] * len(self.descriptions)

    def __len__(self):
        return len(self.image_paths)

    def _generate_probabilities(self, count):
        """
        [1/2, 1/4, 1/8, ... 1/2^(count - 1), 1/2^(count - 1)]
        or [1]
        """

        probabilities = [1 / 2 ** (i + 1) for i in range(count)]
        if len(probabilities) > 1:
            probabilities[-1] = probabilities[-2]
        else:
            probabilities = [1]

        return probabilities

    def __getitem__(self, idx):
        probabilites = self._generate_probabilities(len(self.image_paths[idx]))
        generating_count = min(self.max_image_stack_size, len(self.image_paths[idx]))

        # Random sample from image paths based on probabilities
        # Choose maximum max_image_stack_size images
        # If there are less images, return only those
        image_paths = self.rnd_state.choice(
            self.image_paths[idx],
            size=generating_count,
            replace=False,
            p=probabilites
        )

        # Load images
        images = []
        for path in image_paths:
            image = torchvision.io.read_image(
                path,
                torchvision.io.ImageReadMode.RGB
            )
            images.append(image)

        # Tokenize label if not already done
        if self.hint_labels[idx] is None:
            self.hint_labels[idx] = self._tokenize(self.hints[idx])
            self.description_labels[idx] = self._tokenize(self.descriptions[idx])

        return images, [self.hint_labels[idx], self.description_labels[idx]]


class CocoDataset(DatasetBase):
    def __init__(
        self,
        root,
        ann_file,
        tokenizer=None,
        max_label_length=512,
        device=None,
        seed=None
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_label_length=max_label_length,
            device=device,
            seed=seed
        )

        self.ds = dset.CocoCaptions(
            root=root,
            annFile=ann_file,
            transform=transforms.PILToTensor()
        )

        self.labels = [[None] * 5 for _ in range(len(self.ds))]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, ann = self.ds[idx]

        # Select random caption
        i = self.rnd_state.randint(0, 5)

        # Tokenize label if not already done
        if self.labels[idx][i] is None:
            self.labels[idx][i] = self._tokenize(ann[i])
        label = self.labels[idx][i]

        return [image], ['', label]


class AppDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        hints=True,
        batch_size=1,
        shuffle=True,
        image_size=448,
        processor=None,
        tokenizer=None,
        max_label_length=512,
        device=None,
        sampler=None
    ):
        self.hints = hints

        self.image_size = image_size
        self.processor = processor

        self.tokenizer = tokenizer
        self.max_label_length = max_label_length

        self.device = device

        if sampler is None:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_fn
            )
        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_fn,
                sampler=sampler
            )

    def collate_fn(self, batch):
        images, labels = zip(*list(batch))
        if self.hints:
            hints, descriptions = zip(*labels)

        if self.processor is not None:
            images = self._image_preprocess(images)
            images = images.to(self.device)

        if self.tokenizer is not None:
            if self.hints:
                hints = self._batch_tokenize(hints)
                descriptions = self._batch_tokenize(descriptions)
            else:
                labels = self._batch_tokenize(labels)
        else:
            try:
                if self.hints:
                    hints = torch.stack(hints, dim=0)
                    descriptions = torch.stack(descriptions, dim=0)
                else:
                    labels = torch.stack(labels, dim=0)
            except TypeError:
                pass

        if self.device is not None:
            if self.hints:
                hints = hints.to(self.device)
                descriptions = descriptions.to(self.device)
            else:
                labels = labels.to(self.device)

        return images, [hints, descriptions] if self.hints else labels

    def _image_preprocess(self, images):
        """Preprocess images using the processor.
        
        Receives a 2D list of images (5D list):
        [
            [image1, image2, image3, ...],
            [image1, image2, image3, ...],
            ...
        ]
        
        Returns a tensor of tensors of batched images (5D tensor):
        [
            [image0_0, image1_0, image2_0, ...],
            [image0_1, image1_1, image2_1, ...],
            ...
        ]
        """

        preprocessed_images = []

        for image_stack in images:
            preprocessed_stack = self.processor(
                image_stack,
                return_tensors="pt"
            ).pixel_values

            preprocessed_images.append(preprocessed_stack)

        images = torch.stack(preprocessed_images, dim=1)

        return images

    def _batch_tokenize(self, labels):
        """Tokenize pad and truncate batch labels."""

        tokenized = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt"
        ).input_ids

        # Append <eos> token if enough space
        if tokenized.shape[1] < self.max_label_length:
            eos_tokens = torch.full(
                (tokenized.shape[0], 1),
                self.tokenizer.eos_token_id
            )
            tokenized = torch.cat((tokenized, eos_tokens), dim=1)

        return tokenized


def load_app_data(
    root,
    data_name,
    tokenizer,
    processor,
    image_size,
    max_image_stack_size,
    max_label_length,
    minibatch_size,
    data_split,
    include_hints,
    device,
    seed,
    local_rank=None
):
    dataset_tokenizer = tokenizer if minibatch_size == 1 else None
    dataset = AppDataset(
        root=root,
        data_name=data_name,
        images_subdir="images",
        max_image_stack_size=max_image_stack_size,
        tokenizer=dataset_tokenizer,
        max_label_length=max_label_length,
        include_hints=include_hints,
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
    train_loader = AppDataLoader(
        train_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        tokenizer=dataloader_tokenizer,
        image_size=image_size,
        device=device,
        sampler=train_sampler
    )

    val_loader = AppDataLoader(
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


def load_coco_data(
    root,
    tokenizer,
    processor,
    image_size,
    max_label_length,
    minibatch_size,
    device,
    seed,
    local_rank=None
):
    dataset_tokenizer = tokenizer if minibatch_size == 1 else None
    train_ds = CocoDataset(
        root=f"{root}/images/train2017",
        ann_file=f"{root}/annotations/captions_train2017.json",
        tokenizer=dataset_tokenizer,
        max_label_length=max_label_length,
        device=device,
        seed=seed
    )
    val_ds = CocoDataset(
        root=f"{root}/images/val2017",
        ann_file=f"{root}/annotations/captions_val2017.json",
        tokenizer=dataset_tokenizer,
        max_label_length=max_label_length,
        device=device,
        seed=seed
    )

    if local_rank is not None:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    else:
        train_sampler = None
        val_sampler = None

    dataloader_tokenizer = tokenizer if minibatch_size != 1 else None
    train_loader = AppDataLoader(
        train_ds,
        hints=False,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        tokenizer=dataloader_tokenizer,
        image_size=image_size,
        device=device,
        sampler=train_sampler
    )

    val_loader = AppDataLoader(
        val_ds,
        hints=False,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        tokenizer=dataloader_tokenizer,
        image_size=image_size,
        device=device,
        sampler=val_sampler
    )

    return train_loader, val_loader
