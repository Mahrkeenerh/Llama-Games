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
            max_image_stack_size=8,
            tokenizer=None,
            max_label_length=512,
            device=None,
            seed=None,
            verbose=False
        ):
        self.max_image_stack_size = max_image_stack_size
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
        ).input_ids[0]

        # Append <eos> token if enough space
        if tokenized.shape[0] < self.max_label_length:
            tokenized = torch.cat(
                (tokenized, torch.tensor([self.tokenizer.eos_token_id]))
            )

        self.labels[idx] = tokenized

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
        if self.labels[idx] is None:
            self._item_tokenize(idx)
        label = self.labels[idx]

        return images, label


class AppDataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=True,
            image_size=448,
            processor=None,
            tokenizer=None,
            max_label_length=512,
            device=None,
            sampler=None
        ):
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

        if self.processor is not None:
            images = self._image_preprocess(images)

        if self.tokenizer is not None:
            labels = self._batch_tokenize(labels)
        else:
            labels = torch.stack(labels, dim=0)

        images = images.to(self.device)
        labels = labels.to(self.device)

        return images, labels

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
