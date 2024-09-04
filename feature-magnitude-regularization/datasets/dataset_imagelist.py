import os
from typing import Optional, Callable, Tuple, List

import torch
import torchvision.datasets as datasets
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose
import numpy as np

from .transforms import ResizeImage, PlaceCrop
from tools.argument_helper import ArgumentHelper


class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification."""

    def __init__(self,
                 root: str,
                 samples_per_class: Optional[int],
                 class_names: List[str],
                 data_list_file: str,
                 full_list_file: str,
                 use_ssl: bool,
                 use_cache: bool = False,
                 exclude_list_file: Optional[str] = None,
                 input_transforms: Optional[List[Callable]] = None,
                 target_transform: Optional[Callable] = None):
        """
        Creates a generic dataset. This should be called by a concrete subclass.
        :param root: Root directory of dataset
        :param samples_per_class: The number of samples from each class to return. If None, ignore it.
        :param class_names: The names of all the classes
        :param data_list_file: File to read the labelled image list from.
        :param full_list_file: File to read the full image list from.
        :param use_ssl: If true, serve up unlabelled samples as well.
        :param exclude_list_file: If provided, we will load the contents of `data_list_file`, but filter out anything
        in this file.
        :param input_transforms:  A list of function/transforms that take in a PIL image and returns a transformed
        version.
        E.g, :class:`torchvision.transforms.RandomCrop`.
        :param target_transform: An optional function/transform that takes in the target and transforms it.
        """
        super().__init__(root, transform=input_transforms[0], target_transform=target_transform)
        self.cached_data = {}
        self.use_cache = ArgumentHelper.check_type(use_cache, bool)
        self.class_names = ArgumentHelper.check_list_of_type(class_names, str)
        self.samples_per_class = ArgumentHelper.check_type_or_none(samples_per_class, int)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.loader = default_loader
        self.data_list_file = ArgumentHelper.check_type(data_list_file, str)
        self.full_list_file = ArgumentHelper.check_type(full_list_file, str)
        self.exclude_list_file = ArgumentHelper.check_type_or_none(exclude_list_file, str)
        self.use_ssl = ArgumentHelper.check_type(use_ssl, bool)
        self.input_transforms = input_transforms

        self.samples: List[Tuple[str, Optional[int]]] = self._load_sample_data()

        self.per_class_indices = []
        """
        The unique index for each sample within its class.
        """

        # Work out the number of samples for each class...
        self.counts_per_class = {}
        for _, label_id in self.samples:
            if label_id is None:
                self.per_class_indices.append(-1)
                continue

            if label_id not in self.counts_per_class:
                self.counts_per_class[label_id] = 0
            self.per_class_indices.append(self.counts_per_class[label_id])
            self.counts_per_class[label_id] += 1

    def _load_sample_data(self) -> List[Tuple[str, Optional[int]]]:
        labelled_samples: List[Tuple[str, int]] = self._parse_data_file(self.data_list_file)

        if self.exclude_list_file is not None:
            excluded_samples: List[str] = [file_name for file_name, _ in self._parse_data_file(self.exclude_list_file)]
            labelled_samples: List[Tuple[str, int]] = self._remove_names(excluded_samples, labelled_samples)

        if self.samples_per_class is not None:
            labelled_samples: List[Tuple[str, int]] = self._trim_to_sample_count(self.samples_per_class,
                                                                                 labelled_samples)

        if self.use_ssl:
            all_samples = {file_name: None for file_name, _ in self._parse_data_file(self.full_list_file)}
            for file_name, label in labelled_samples:
                all_samples[file_name] = label

            finished_data: List[Tuple[str, int]] = []
            for file_name in all_samples:
                finished_data.append((file_name, all_samples[file_name]))
            return finished_data
        else:
            return labelled_samples

    @staticmethod
    def _remove_names(exclude_names: List[str], all_samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Removes any samples with a name in `exclude_names`.
        :param exclude_names: The names to remove.
        :param all_samples: The full list of samples to filter.
        :returns The filtered list of samples.
        """
        samples = []

        for path, image_label in all_samples:
            if path not in exclude_names:
                samples.append((path, image_label))

        return samples

    @staticmethod
    def _trim_to_sample_count(sample_count: int, all_samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Trims any excess samples so that all classes have `sample_count` samples.
        :param sample_count: The number of samples we want.
        :param all_samples: The full list of samples to filter.
        :returns The filtered list of samples.
        """
        samples = []

        counts = {}
        for path, image_label in all_samples:
            if image_label not in counts:
                counts[image_label] = 0

            if counts[image_label] < sample_count:
                counts[image_label] += 1

                samples.append((path, image_label))

        return samples

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int, int, int]:
        """
        Gets a data item for the set.
        :parameter index: The index of the data to return.
        :returns A tuple of the input image, the target class, the image index and the unique index with its class.
        """
        path, target = self.samples[index]
        if self.use_cache:
            if index not in self.cached_data:
                self.cached_data[index] = self.loader(path)

            img = self.cached_data[index]
        else:
            try:
                img = self.loader(path)
            except FileNotFoundError as ex:
                print(f'File {path} not found.')
                raise ex

        per_sample_index = self.per_class_indices[index]

        if self.input_transforms is not None:
            images = [transform(img) for transform in self.input_transforms]
        else:
            images = [img]
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        if target is None:
            target = -1
            per_sample_index = -1

        return images, target, index, per_sample_index

    def get_input_image(self, index: int) -> Tuple[np.ndarray, str]:
        """
        Gets an input image corresponding to a given index, as an RGB Numpy array.
        :param index: The index to of the image to fetch.
        :return: Tuple containing The image as an RGB Numpy array and the name of the original image file.
        """
        assert index < len(self), "Index should never get more than the max allowed."

        img_path, _ = self.samples[index]

        resize_size = 256
        crop_size = 224
        start_center = (resize_size - crop_size - 1) / 2
        transform = Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center)
        ])

        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        return np.asarray(img), os.path.basename(img_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """
        Parses a list of files that contain the data to serve up.
        :parameter file_name: The path to the file containing the file data.
        :returns A list of (file path, target) tuples.
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def class_count(self) -> int:
        """Number of classes in the dataset."""
        return len(self.class_names)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented
