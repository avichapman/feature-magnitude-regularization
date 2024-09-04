import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize

from .transforms import imagenet_mean, imagenet_std, TransformTest, ResizeImage, PlaceCrop
from tools.argument_helper import ArgumentHelper


class DatasetINaturalist(Dataset):
    """
    Manages iNaturalist 2021 Mini dataset.
    """

    # noinspection PyUnusedLocal
    def __init__(self,
                 list_root: str,
                 files_root: str,
                 is_train: bool,
                 is_test: bool,
                 use_ssl: bool,
                 trial_index: int,
                 subset: Optional[str] = 'Passeriformes',
                 single_test_transform: bool = False,
                 label_ratio: Optional[int] = None,
                 samples_per_class: Optional[int] = None):
        """
        Creates a new dataset.
        :param list_root: Root directory of dataset lists of files.
        :param files_root: Root directory of the files to load. If the lists contain relative paths, they will be
        relative to this path.
        :param is_train: If true, use training data.
        :param is_test: If true, use testing data.
        :param use_ssl: If true, serve up unlabelled samples as well.
        :param trial_index: The index of this trial. Used to make the random splits repeatable.
        :param single_test_transform: If true, only use a single evaluation transform.
        :param label_ratio: Integer from 1-100 describing the percentage of labelled data to serve up. If None, defer
        to `samples_per_class`.
        :param samples_per_class: The number of samples from each class to return. If None, defer to `label_ratio`.
        """
        super().__init__()

        self._data_dir = Path(ArgumentHelper.check_type(files_root, str)).resolve()
        self.is_training_data = ArgumentHelper.check_type(is_train, bool)
        self.label_ratio = ArgumentHelper.check_type_or_none(label_ratio, int)
        self.samples_per_class = ArgumentHelper.check_type_or_none(samples_per_class, int)
        self.single_test_transform = ArgumentHelper.check_type(single_test_transform, bool)
        self.use_ssl = ArgumentHelper.check_type(use_ssl, bool)
        self.trial_index = ArgumentHelper.check_type(trial_index, int)
        self.max_samples_per_class = 50 if self.is_training_data else 10
        self.subset = ArgumentHelper.check_type_or_none(subset, str)

        if self.samples_per_class == 0:
            self.samples_per_class = None

        if self.label_ratio is None and self.samples_per_class is None:
            self.label_ratio = 100

        if self.label_ratio is not None:
            assert 0 < self.label_ratio <= 100, "Label ratio must be between 1-100."

        # Use samples per class to ensure equality...
        if self.label_ratio < 100:
            self.samples_per_class = (self.label_ratio * self.max_samples_per_class) // 100
            self.label_ratio = 100

        np.random.seed(self.trial_index)

        self.transforms: List[Callable] = self._get_transforms()

        self.feature_count = 3

        data = self._load_data()
        self.filtered_data: List[Tuple[Path, int]] = self._filter_data(data)

        # Work out the number of samples for each class...
        self.counts_per_class = {}
        for _, label_id in self.filtered_data:
            if label_id is None:
                continue

            if label_id not in self.counts_per_class:
                self.counts_per_class[label_id] = 0
            self.counts_per_class[label_id] += 1

        category_data, category_id_map = self._get_category_data()
        self.class_count = len(category_data)
        self.class_names: List[str] = list(category_data.values())
        self.class_id_mappings: Dict[int, int] = category_id_map

        # Ensure that assumptions about category IDs are met...
        category_ids = list(self.class_id_mappings.values())

        assert max(category_ids) == self.class_count - 1, "There must be missing or offset category IDs."
        assert min(category_ids) == 0, "There must be missing or offset category IDs."

    def _get_category_data(self) -> Tuple[Dict[int, str], Dict[int, int]]:
        """
        Builds category data.
        :return: A tuple containing:
            - A dictionary of category IDs and names
            - A dictionary of global category ids to subset ids
        """
        category_data: Dict[int, str] = {}
        category_id_map: Dict[int, int] = {}
        for file_path, cat_id in self.filtered_data:
            category_name = "_".join(file_path.parent.name.split("_")[1:])
            if cat_id not in category_data:
                category_data[cat_id] = category_name
                category_id_map[cat_id] = len(category_id_map)

        return category_data, category_id_map

    def _load_data(self) -> List[Tuple[Path, int]]:
        """
        Loads the data from the appropriate directory, along with its category.
        :return: A list of Path and integer pairs.
        """
        if self.is_training_data:
            data_path = self._data_dir / 'train'
        else:
            data_path = self._data_dir / 'val'

        data: List[Tuple[Path, int]] = []
        for sub_dir in data_path.glob("*"):
            if sub_dir.is_dir():
                category_id = int(sub_dir.name.split("_")[0])
                for file_path in sub_dir.glob("*.jpg"):
                    data.append((file_path, category_id))
        return data

    def _filter_to_subset(self, unfiltered_data: List[Tuple[Path, int]]) -> List[Tuple[Path, int]]:
        if self.subset is not None:
            # Reduce to the selected subset before applying the rest of the filter...
            filtered_data: List[Tuple[Path, int]] = []
            for file_path, cat_id in unfiltered_data:
                if str(file_path).find(self.subset) > -1:
                    filtered_data.append((file_path, cat_id))
            return filtered_data
        else:
            return unfiltered_data

    def _filter_data(self, unfiltered_data: List[Tuple[Path, int]]) -> List[Tuple[Path, int]]:
        """
        Filters a list of Path and category ID pairs to the desired subset.
        :return: The filtered list.
        """
        unfiltered_data = self._filter_to_subset(unfiltered_data)

        filtered_data: List[Tuple[Path, int]] = []
        if self.samples_per_class is not None:
            counts_class = {}
            for i in range(len(unfiltered_data)):
                file_path, cat_id = unfiltered_data[i]

                if cat_id not in counts_class:
                    counts_class[cat_id] = 0

                if counts_class[cat_id] < self.samples_per_class:
                    filtered_data.append((file_path, cat_id))
                    counts_class[cat_id] += 1
                elif self.use_ssl:
                    filtered_data.append((file_path, -1))
        else:
            indices = np.random.choice(range(len(unfiltered_data)),
                                       size=len(unfiltered_data) * self.label_ratio // 100,
                                       replace=False)

            for i in range(len(unfiltered_data)):
                file_path, cat_id = unfiltered_data[i]

                if i in indices:
                    filtered_data.append((file_path, cat_id))
                elif self.use_ssl:
                    filtered_data.append((file_path, -1))

        return filtered_data

    def _get_transforms(self) -> List[Callable]:
        """
        Returns the tensor transform to apply to the dataset.
        """
        if self.is_training_data:
            return [Compose([
                Resize((256, 256)),
                RandomHorizontalFlip(),
                RandomResizedCrop(224),
                ToTensor(),
                Normalize(imagenet_mean, imagenet_std)])]
        else:
            if self.single_test_transform:
                return [TransformTest(mean=imagenet_mean,
                                      std=imagenet_std)['test9']]
            else:
                return list(TransformTest(mean=imagenet_mean,
                                          std=imagenet_std).values())

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            index, int: Index.
        Returns:
            - Sample
            - Class Index
            - Unique Sample Index
        """
        assert index < len(self), "Index should never get more than the max allowed."

        image_path, class_id = self.filtered_data[index]
        label_id = self.class_id_mappings[class_id]

        # Load it as an image and then apply any transforms...
        img = Image.open(image_path).convert('RGB')
        images = [transform(img) for transform in self.transforms]

        # To maintain compatibility with other datasets
        if self.is_training_data:
            images = [images]

        return images, label_id, index

    def get_input_image(self, index: int) -> Tuple[np.ndarray, str]:
        """
        Gets an input image corresponding to a given index, as an RGB Numpy array.
        :param index: The index to of the image to fetch.
        :return: Tupel containing The image as an RGB Numpy array and the name of the original image file.
        """
        assert index < len(self), "Index should never get more than the max allowed."

        resize_size = 256
        crop_size = 224
        start_first = 0
        transform = Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_first, start_first)
        ])

        img_path, _ = self.filtered_data[index]
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        return np.asarray(img), os.path.basename(img_path)


if __name__ == "__main__":
    data_dir_ = '../Datasets/iNaturalist/'
    ds = DatasetINaturalist(files_root=data_dir_,
                            list_root=r'',
                            is_train=False,
                            is_test=False,
                            use_ssl=False,
                            trial_index=0,
                            label_ratio=100)
    print("Total Samples:", len(ds))
    print("Class Count:", ds.class_count)
    # print("Class Names:", ds.class_names)

    class_counts_ = {}

    for index_ in range(len(ds.filtered_data)):
        _, class_id_ = ds.filtered_data[index_]
        if class_id_ not in class_counts_:
            class_counts_[class_id_] = 0

        class_counts_[class_id_] += 1

    # print('Class Counts:')
    # for class_id_ in class_counts_:
    #     print(f"{class_id_}: {class_counts_[class_id_]}")
    #
    # print("Example Class Name:")
    # print(ds.class_names[0])
    # print(ds.class_names[1])

    ds_loader = DataLoader(ds,
                           num_workers=0,
                           pin_memory=True,
                           shuffle=True,
                           batch_size=1)

    for index_, (img_, class_id_, data_index_) in enumerate(ds_loader):
        print(index_, data_index_, class_id_)
        break
