import os
from typing import Optional

from torch.utils.data import DataLoader

from tools.argument_helper import ArgumentHelper
from .dataset_imagelist import ImageList
from .transforms import TransformTrain, imagenet_mean, imagenet_std, TransformTest


class DatasetStanfordCars(ImageList):
    """
    `The Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    """

    download_list = [
        ("image_list", "image_list.zip", ""),
        ("train", "train.tgz", ""),
        ("test", "test.tgz", ""),
    ]

    image_list = {
        "train": "image_list/train_100.txt",
        "train100": "image_list/train_100.txt",
        "train50": "image_list/train_50.txt",
        "train30": "image_list/train_30.txt",
        "train15": "image_list/train_15.txt",
        "train10": "image_list/train_10.txt",
        "train1019630": "image_list/train_1019630.txt",
        "train1519630": "image_list/train_1519630.txt",
        "train3019630": "image_list/train_3019630.txt",
        "train5019630": "image_list/train_5019630.txt",
        "train1019650": "image_list/train_1019650.txt",
        "train1519650": "image_list/train_1519650.txt",
        "train3019650": "image_list/train_3019650.txt",
        "train5019650": "image_list/train_5019650.txt",
        "train10196100": "image_list/train_10196100.txt",
        "train15196100": "image_list/train_15196100.txt",
        "train30196100": "image_list/train_30196100.txt",
        "train50196100": "image_list/train_50196100.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt",
    }
    CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20',
               '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
               '38', '39', '40',
               '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57',
               '58', '59', '60',
               '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77',
               '78', '79', '80',
               '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97',
               '98', '99', '100',
               '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115',
               '116', '117', '118', '119', '120',
               '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135',
               '136', '137', '138', '139', '140',
               '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155',
               '156', '157', '158', '159', '160',
               '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175',
               '176', '177', '178', '179', '180',
               '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195',
               '196']

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
        Creates a new dataset.
        :argument list_root: The path to the dataset data.
        :argument files_root: Root directory of the files to load. Not applicable to this dataset, but included for
        compatibility.
        :param is_train: If true, use training data.
        :param is_test: If true, use testing data.
        :param use_ssl: If true, serve up unlabelled samples as well.
        :param trial_index: The index of this trial. Used to make the random splits repeatable.
        :param single_test_transform: If true, only use a single evaluation transform.
        :argument label_ratio: Integer from 1-100 describing the percentage of labelled data to serve up. If None, defer
        to `samples_per_class`.
        :argument samples_per_class: The number of samples from each class to return. If None, defer to `label_ratio`.
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
            class_names=DatasetStanfordCars.CLASSES,
            data_list_file=data_list_file,
            full_list_file=full_list_file,
            use_ssl=use_ssl,
            use_cache=True,
            input_transforms=list(transforms.values()))


if __name__ == "__main__":
    source_dir_ = r'D:\Datasets\StanfordCars'
    ds = DatasetStanfordCars(files_root=source_dir_,
                             list_root=os.path.join('..', 'CARS'),
                             is_train=True,
                             is_test=False,
                             trial_index=0,
                             use_ssl=True,
                             label_ratio=10)

    ds_loader = DataLoader(ds,
                           num_workers=1,
                           pin_memory=True,
                           shuffle=True,
                           batch_size=50)

    for index_, (img_, class_id_, data_index_) in enumerate(ds_loader):
        print(index_, data_index_, class_id_)
        break
