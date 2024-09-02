import os
from typing import Dict, Tuple, Union, Optional

from torch.utils.data import DataLoader

from dataset_cub_200 import CUB200
from dataset_fgvc_aircraft import DatasetFgvcAircraft
from dataset_inaturalist import DatasetINaturalist
from dataset_stanford_cars import DatasetStanfordCars

FgvcDataset = Union[CUB200, DatasetFgvcAircraft, DatasetStanfordCars]


def load_data(
        list_root: str,
        files_root: str,
        dataset_name: str,
        use_ssl: bool,
        trial_index: int,
        label_ratio: Optional[int],
        samples_per_class: Optional[int],
        batch_size: int,
        single_test_transform: bool = False,
        num_workers: int = 4,
        load_test: bool = False) -> Tuple[Dict[str, DataLoader], Dict[str, FgvcDataset]]:
    """
    Creates the data loaders needed for training and testing.
    :param list_root: Root directory of dataset lists of files. The data will be in a subdirectory with the name
    supplied in `dataset_name`.
    :param files_root: Root directory of the files to load. If the lists contain relative paths, they will be relative
    to this path.
    :param dataset_name: Name of the dataset.
    :param use_ssl: If true, serve up unlabelled samples as well.
    :param trial_index: The index of this trial. Used to make the random splits repeatable.
    :param label_ratio: Integer from 1-100 describing the percentage of labelled data to serve up. If None, defer
    to `samples_per_class`.
    :param samples_per_class: The number of samples from each class to return. If None, defer to `label_ratio`.
    :param batch_size: Number of samples per mini-batch in training mode.
    :param single_test_transform: If true, only use a single evaluation transform.
    :param num_workers: Number of workers to load data.
    :param load_test: If true, load the test dataset. Otherwise, load the training and validation dataset.
    :return: A tuple containing:
    - A dictionary of data loaders, indexed by use. Includes 'train' and 10 'test' configs.
    - A dictionary of datasets, indexed by use. Includes 'train' and 10 'test' configs.
    """
    dataset_list_path = os.path.join(list_root, dataset_name)
    if dataset_name.upper() == 'CUB200':
        dataset = CUB200
    elif dataset_name.upper() == 'FGVCAC':
        dataset = DatasetFgvcAircraft
    elif dataset_name.upper() == 'CARS':
        dataset = DatasetStanfordCars
    elif dataset_name.upper() == 'INATURALIST':
        dataset = DatasetINaturalist
    else:
        raise ValueError(f"Encountered Unknown Dataset: {dataset_name}")

    if load_test:
        datasets = {"test": dataset(list_root=dataset_list_path,
                                    files_root=files_root,
                                    is_train=False,
                                    is_test=True,
                                    trial_index=trial_index,
                                    use_ssl=False,
                                    single_test_transform=single_test_transform,
                                    label_ratio=100)
                    }

        dataset_loaders = {'test': DataLoader(datasets['test'],
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False,
                                              num_workers=num_workers)
                           }
    else:
        datasets = {"train": dataset(list_root=dataset_list_path,
                                     files_root=files_root,
                                     is_train=True,
                                     is_test=False,
                                     trial_index=trial_index,
                                     use_ssl=use_ssl,
                                     label_ratio=label_ratio,
                                     samples_per_class=samples_per_class),
                    "val": dataset(list_root=dataset_list_path,
                                   files_root=files_root,
                                   is_train=False,
                                   is_test=False,
                                   trial_index=trial_index,
                                   use_ssl=False,
                                   single_test_transform=single_test_transform,
                                   label_ratio=100)
                    }

        dataset_loaders = {'train': DataLoader(datasets['train'],
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               shuffle=True,
                                               num_workers=num_workers),
                           'val': DataLoader(datasets['val'],
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             shuffle=False,
                                             num_workers=num_workers)
                           }

    return dataset_loaders, datasets
