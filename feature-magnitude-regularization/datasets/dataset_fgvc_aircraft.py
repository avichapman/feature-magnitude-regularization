from typing import Optional
import os.path

from torch.utils.data import DataLoader

from ..tools.argument_helper import ArgumentHelper
from dataset_imagelist import ImageList
from transforms import TransformTrain, imagenet_mean, imagenet_std, TransformTest


class DatasetFgvcAircraft(ImageList):
    """
    `FVGC-Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    """

    image_list = {
        "train": "image_list/train_100.txt",
        "train100": "image_list/train_100.txt",
        "train50": "image_list/train_50.txt",
        "train30": "image_list/train_30.txt",
        "train15": "image_list/train_15.txt",
        "train10": "image_list/train_10.txt",
        "train1010030": "image_list/train_1010030.txt",
        "train1510030": "image_list/train_1510030.txt",
        "train3010030": "image_list/train_3010030.txt",
        "train5010030": "image_list/train_5010030.txt",
        "train1010050": "image_list/train_1010050.txt",
        "train1510050": "image_list/train_1510050.txt",
        "train3010050": "image_list/train_3010050.txt",
        "train5010050": "image_list/train_5010050.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt",
    }
    CLASSES = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800',
               '737-900', '747-100', '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300',
               '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200',
               'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12',
               'BAE 146-200', 'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200',
               'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'Cessna 560', 'Challenger 600', 'DC-10',
               'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400',
               'Dornier 328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600',
               'Eurofighter Typhoon', 'F-16A-B', 'F-A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50',
               'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011', 'MD-11',
               'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340',
               'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']

    # noinspection PyUnusedLocal
    def __init__(self,
                 list_root: str,
                 files_root: str,
                 is_train: bool,
                 is_test: bool,
                 use_ssl: bool,
                 trial_index: int,
                 single_test_transform: bool = False,
                 label_ratio: Optional[int] = None,
                 samples_per_class: Optional[int] = None):
        """
        Create a new FGVCAC Aircraft dataset.
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
        self.use_train_data = ArgumentHelper.check_type(is_train, bool)
        self.single_test_transform = ArgumentHelper.check_type(single_test_transform, bool)
        self.label_ratio = ArgumentHelper.check_type_or_none(label_ratio, int)
        self.samples_per_class = ArgumentHelper.check_type_or_none(samples_per_class, int)
        self.trial_index = ArgumentHelper.check_type(trial_index, int)

        if self.samples_per_class == 0:
            self.samples_per_class = None

        if self.label_ratio is None and self.samples_per_class is None:
            self.label_ratio = 100

        if self.label_ratio is not None:
            assert 0 < self.label_ratio <= 100, "Label ratio must be between 1-100."

        if self.use_train_data:
            list_name = 'train' + str(label_ratio)
            assert list_name in self.image_list
            data_list_file = os.path.join(list_root, self.image_list[list_name])
            full_list_file = os.path.join(list_root, self.image_list['train100'])
            transforms = {'train': TransformTrain()}
        else:
            data_list_file = os.path.join(list_root, self.image_list['test'])
            full_list_file = data_list_file

            if self.single_test_transform:
                transforms = {'test': TransformTest(mean=imagenet_mean, std=imagenet_std)['test9']}
            else:
                transforms = TransformTest(mean=imagenet_mean, std=imagenet_std)

        super().__init__(
            root=files_root,
            samples_per_class=self.samples_per_class,
            class_names=DatasetFgvcAircraft.CLASSES,
            data_list_file=data_list_file,
            full_list_file=full_list_file,
            use_ssl=use_ssl,
            use_cache=True,
            input_transforms=list(transforms.values()))


if __name__ == "__main__":
    source_dir_ = 'D:\\Datasets\\FGVC_Tsinghua'
    ds = DatasetFgvcAircraft(files_root=source_dir_,
                             list_root=os.path.join('..', 'FGVCAC'),
                             is_train=True,
                             is_test=False,
                             trial_index=0,
                             use_ssl=True,
                             label_ratio=10)
    ds_eval = DatasetFgvcAircraft(files_root=source_dir_,
                                  list_root=os.path.join('..', 'FGVCAC'),
                                  is_train=False,
                                  is_test=False,
                                  trial_index=0,
                                  use_ssl=False,
                                  label_ratio=100)

    for eval_name, _ in ds_eval.samples:
        if eval_name in [name for name, _ in ds.samples]:
            print(eval_name)

    ds_loader = DataLoader(ds,
                           num_workers=1,
                           pin_memory=True,
                           shuffle=True,
                           batch_size=2)

    for index_, (img_, class_id_, data_index_) in enumerate(ds_loader):
        print(index_, data_index_, class_id_)
        break
