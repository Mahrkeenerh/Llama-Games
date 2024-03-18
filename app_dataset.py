import json
import os
import random

import numpy as np
import torch
import torchvision


class AppDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            data_name,
            images_subdir,
            image_stack_size=8,
            tokenizer=None,
            max_label_length=512,
            device=None,
            seed=None,
            verbose=False
        ):
        self.image_stack_size = image_stack_size
        self.tokenizer = tokenizer
        self.max_label_length = max_label_length

        seed = seed or random.randint(0, 2 ** 32 - 1)
        self.rnd_state = np.random.RandomState(seed)

        self.device = device
        self.verbose = verbose

        if self.verbose:
            print("Loading datafile...")

        self._load_datafile(root, data_name, images_subdir)

    def _load_datafile(self, root, data_name, images_subdir):
        data_json = json.load(open(os.path.join(root, data_name), "r", encoding="utf-8"))
        dataset_raw = {
            app["appId"]: {"image_paths": [], "description": app["description"]}
            for app in data_json
        }

        # Add image paths
        for image in os.listdir(os.path.join(root, images_subdir)):
            app_id = "_".join(image.split("_")[:-1])
            if app_id in dataset_raw:
                dataset_raw[app_id]["image_paths"].append(os.path.join(root, images_subdir, image))
            else:
                if self.verbose:
                    print("Not found", image)

        self.image_paths = []
        self.descriptions = []

        # Extract image paths and descriptions into separate lists
        for app_id, app in dataset_raw.items():
            if len(app["image_paths"]) > 0:
                self.image_paths.append(app["image_paths"])
                self.descriptions.append(app["description"])
            else:
                if self.verbose:
                    print("No images for", app_id)

        self.labels =  [None] * len(self.descriptions)

    def _item_tokenize(self, idx):
        if self.tokenizer is None:
            self.labels[idx] = self.descriptions[idx]
            return

        tokenized = self.tokenizer(
            self.descriptions[idx],
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt"
        ).input_ids
        self.labels[idx] = tokenized[0]

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
        image_paths = []

        # Keep randomly adding images - if less, all will be included at least once
        while len(image_paths) < self.image_stack_size:
            generating_count = min(self.image_stack_size - len(image_paths), len(self.image_paths[idx]))
            image_paths.extend(self.rnd_state.choice(
                self.image_paths[idx],
                size=generating_count,
                replace=False,
                p=probabilites
            ))

        if self.labels[idx] is None:
            self._item_tokenize(idx)
        label = self.labels[idx]

        return image_paths, label


class AppDataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=True,
            image_size=448,
            do_prepcocess=True,
            tokenizer=None,
            max_label_length=512,
            device=None,
            sampler=None
        ):
        self.image_size = image_size
        self.do_prepcocess = do_prepcocess

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

        images = self._load_images(images)
        images = self._image_crop(images)
        images = self._image_pad(images)

        if self.do_prepcocess:
            images = self._image_normalize(images)

        if self.tokenizer is not None:
            labels = self._batch_tokenize(labels)
        else:
            labels = torch.stack(labels, dim=0)

        images = images.to(self.device)
        labels = labels.to(self.device)

        return images, labels

    def _batch_tokenize(self, labels):
        """Tokenize pad and truncate batch labels."""

        tokenized = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt"
        )

        return tokenized["input_ids"]

    def _load_images(self, image_paths):
        """Receives a 2D list of image paths:
        [
            [path1, path2, path3, ...],
            [path1, path2, path3, ...],
            ...
        ]

        Returns a 2D lsit of images (5D list):
        [
            [image1, image2, image3, ...],
            [image1, image2, image3, ...],
            ...
        ]
        """

        images = []

        for path_stack in image_paths:
            image_stack = []

            for path in path_stack:
                image = torchvision.io.read_image(
                    path,
                    torchvision.io.ImageReadMode.RGB
                )
                image_stack.append(image)

            images.append(image_stack)

        return images

    def _image_crop(self, images):
        """Randomly crop images to self.image_size x self.image_size.

        Receives a 2D list of images (5D list):
        [
            [image1, image2, image3, ...],
            [image1, image2, image3, ...],
            ...
        ]

        Returns the same data format.
        """

        cropped_images = []

        for image_stack in images:
            cropped_stack = []

            for image in image_stack:
                h = image.shape[-2]
                w = image.shape[-1]

                h_start = random.randint(0, max(h - self.image_size, 0))
                w_start = random.randint(0, max(w - self.image_size, 0))

                h_end = min(h_start + self.image_size, h)
                w_end = min(w_start + self.image_size, w)

                cropped_image = image[..., h_start:h_end, w_start:w_end]
                cropped_stack.append(cropped_image)

            cropped_images.append(cropped_stack)

        return cropped_images

    def _image_pad(self, images):
        """Receives a 2D list of images (5D list):
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

        padded_images = []

        for image_stack in images:
            padded_stack = []

            for image in image_stack:
                h = image.shape[-2]
                w = image.shape[-1]
                h_start = (self.image_size - h) // 2
                w_start = (self.image_size - w) // 2

                padded_image = torch.nn.functional.pad(image, (
                    w_start,
                    self.image_size - w - w_start,
                    h_start,
                    self.image_size - h - h_start
                ), value=0)
                padded_stack.append(padded_image)

            padded_images.append(padded_stack)

        images = torch.stack([torch.stack(image_stack, dim=0) for image_stack in padded_images], dim=1)

        return images

    def _image_normalize(self, images):
        # Mean 0.5 and std 0.5, same as pretrained ViTImageProcessor
        images = images / 255.0
        images = (images - 0.5) / 0.5

        return images
