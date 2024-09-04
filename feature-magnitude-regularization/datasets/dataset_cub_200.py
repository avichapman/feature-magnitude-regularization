from typing import Optional, List, Dict
import os.path

from torch.utils.data import DataLoader

from tools.argument_helper import ArgumentHelper
from .dataset_imagelist import ImageList
from .transforms import TransformTrain, imagenet_mean, imagenet_std, TransformTest


class CUB200(ImageList):
    """
    `Caltech-UCSD Birds-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
    """
    image_list = {
        "train": "image_list/train_100.txt",
        "train100": "image_list/train_100.txt",
        "train75": "image_list/train_75.txt",
        "train50": "image_list/train_50.txt",
        "train30": "image_list/train_30.txt",
        "train15": "image_list/train_15.txt",
        "train10": "image_list/train_10.txt",
        "train1020030": "image_list/train_1020030.txt",
        "train1520030": "image_list/train_1520030.txt",
        "train3020030": "image_list/train_3020030.txt",
        "train5020030": "image_list/train_5020030.txt",
        "train1020050": "image_list/train_1020050.txt",
        "train1520050": "image_list/train_1520050.txt",
        "train3020050": "image_list/train_3020050.txt",
        "train5020050": "image_list/train_5020050.txt",
        "train10200100": "image_list/train_10200100.txt",
        "train15200100": "image_list/train_15200100.txt",
        "train30200100": "image_list/train_30200100.txt",
        "train50200100": "image_list/train_50200100.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt"
    }
    CLASSES = ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani',
               '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet',
               '009.Brewer_Blackbird', '010.Red_winged_Blackbird', '011.Rusty_Blackbird', '012.Yellow_headed_Blackbird',
               '013.Bobolink', '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Bunting', '017.Cardinal',
               '018.Spotted_Catbird', '019.Gray_Catbird', '020.Yellow_breasted_Chat', '021.Eastern_Towhee',
               '022.Chuck_will_Widow', '023.Brandt_Cormorant', '024.Red_faced_Cormorant', '025.Pelagic_Cormorant',
               '026.Bronzed_Cowbird', '027.Shiny_Cowbird', '028.Brown_Creeper', '029.American_Crow', '030.Fish_Crow',
               '031.Black_billed_Cuckoo', '032.Mangrove_Cuckoo', '033.Yellow_billed_Cuckoo',
               '034.Gray_crowned_Rosy_Finch', '035.Purple_Finch', '036.Northern_Flicker', '037.Acadian_Flycatcher',
               '038.Great_Crested_Flycatcher', '039.Least_Flycatcher', '040.Olive_sided_Flycatcher',
               '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '043.Yellow_bellied_Flycatcher',
               '044.Frigatebird', '045.Northern_Fulmar', '046.Gadwall', '047.American_Goldfinch',
               '048.European_Goldfinch', '049.Boat_tailed_Grackle', '050.Eared_Grebe',
               '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '054.Blue_Grosbeak',
               '055.Evening_Grosbeak', '056.Pine_Grosbeak', '057.Rose_breasted_Grosbeak', '058.Pigeon_Guillemot',
               '059.California_Gull', '060.Glaucous_winged_Gull', '061.Heermann_Gull', '062.Herring_Gull',
               '063.Ivory_Gull', '064.Ring_billed_Gull', '065.Slaty_backed_Gull', '066.Western_Gull',
               '067.Anna_Hummingbird', '068.Ruby_throated_Hummingbird', '069.Rufous_Hummingbird', '070.Green_Violetear',
               '071.Long_tailed_Jaeger', '072.Pomarine_Jaeger', '073.Blue_Jay', '074.Florida_Jay', '075.Green_Jay',
               '076.Dark_eyed_Junco', '077.Tropical_Kingbird', '078.Gray_Kingbird', '079.Belted_Kingfisher',
               '080.Green_Kingfisher', '081.Pied_Kingfisher', '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher',
               '084.Red_legged_Kittiwake', '085.Horned_Lark', '086.Pacific_Loon', '087.Mallard',
               '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser', '091.Mockingbird',
               '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole',
               '096.Hooded_Oriole', '097.Orchard_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican',
               '101.White_Pelican', '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit',
               '105.Whip_poor_Will', '106.Horned_Puffin', '107.Common_Raven', '108.White_necked_Raven',
               '109.American_Redstart', '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike',
               '113.Baird_Sparrow', '114.Black_throated_Sparrow', '115.Brewer_Sparrow', '116.Chipping_Sparrow',
               '117.Clay_colored_Sparrow', '118.House_Sparrow', '119.Field_Sparrow', '120.Fox_Sparrow',
               '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '124.Le_Conte_Sparrow',
               '125.Lincoln_Sparrow', '126.Nelson_Sharp_tailed_Sparrow', '127.Savannah_Sparrow', '128.Seaside_Sparrow',
               '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow', '132.White_crowned_Sparrow',
               '133.White_throated_Sparrow', '134.Cape_Glossy_Starling', '135.Bank_Swallow', '136.Barn_Swallow',
               '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager', '140.Summer_Tanager', '141.Artic_Tern',
               '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern', '146.Forsters_Tern',
               '147.Least_Tern', '148.Green_tailed_Towhee', '149.Brown_Thrasher', '150.Sage_Thrasher',
               '151.Black_capped_Vireo', '152.Blue_headed_Vireo', '153.Philadelphia_Vireo', '154.Red_eyed_Vireo',
               '155.Warbling_Vireo', '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', '158.Bay_breasted_Warbler',
               '159.Black_and_white_Warbler', '160.Black_throated_Blue_Warbler', '161.Blue_winged_Warbler',
               '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler', '165.Chestnut_sided_Warbler',
               '166.Golden_winged_Warbler', '167.Hooded_Warbler', '168.Kentucky_Warbler', '169.Magnolia_Warbler',
               '170.Mourning_Warbler', '171.Myrtle_Warbler', '172.Nashville_Warbler', '173.Orange_crowned_Warbler',
               '174.Palm_Warbler', '175.Pine_Warbler', '176.Prairie_Warbler', '177.Prothonotary_Warbler',
               '178.Swainson_Warbler', '179.Tennessee_Warbler', '180.Wilson_Warbler', '181.Worm_eating_Warbler',
               '182.Yellow_Warbler', '183.Northern_Waterthrush', '184.Louisiana_Waterthrush', '185.Bohemian_Waxwing',
               '186.Cedar_Waxwing', '187.American_Three_toed_Woodpecker', '188.Pileated_Woodpecker',
               '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker', '191.Red_headed_Woodpecker',
               '192.Downy_Woodpecker', '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '196.House_Wren',
               '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat']

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
                 reserve_ratio: int = 0,
                 use_reserve: bool = False,
                 samples_per_class: Optional[int] = None):
        """
        Create a new CUB200 dataset.
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
        :param reserve_ratio: Integer from 0-99 describing the percentage of the labelled data to reserve. If non-zero,
        this amount will NOT be served for training unless `use_reserve` is TRUE, in which case this is the ONLY data
        being served.
        :param use_reserve: If true, serve up the reserved data instead of the main data.
        :param samples_per_class: The number of samples from each class to return. If None, defer to `label_ratio`.
        """
        self.use_train_data = ArgumentHelper.check_type(is_train, bool)
        self.single_test_transform = ArgumentHelper.check_type(single_test_transform, bool)
        self.label_ratio = ArgumentHelper.check_type_or_none(label_ratio, int)
        self.samples_per_class = ArgumentHelper.check_type_or_none(samples_per_class, int)
        self.trial_index = ArgumentHelper.check_type(trial_index, int)
        self.reserve_ratio = ArgumentHelper.check_type(reserve_ratio, int)
        self.use_reserve = ArgumentHelper.check_type(use_reserve, bool)

        if self.samples_per_class == 0:
            self.samples_per_class = None

        if self.label_ratio is None and self.samples_per_class is None:
            self.label_ratio = 100

        supported_ratios = [10, 15, 30, 50, 100]
        if self.label_ratio is not None:
            assert self.label_ratio in supported_ratios, f"Label ratio must be one of {supported_ratios}."

        supported_reserves: Dict[int, List[int]] = {
            10: [0],
            15: [0, 5, 10],
            30: [0, 10, 15],
            50: [0, 10, 15, 20, 30],
            100: [0, 10, 15, 30, 50]
        }
        assert self.reserve_ratio in supported_reserves[self.label_ratio], \
            f"Reserve for label ratio {self.label_ratio} must be one of {supported_reserves[self.label_ratio]}"
        assert not (self.use_reserve and self.reserve_ratio == 0), "Cannot use a reserve of zero."

        if self.use_train_data:
            list_name = 'train' + str(label_ratio)
            data_list_file = os.path.join(list_root, self.image_list[list_name])

            if self.reserve_ratio > 0:
                reserve_list_file = None
                exclude_reserve_list = None

                # Specifies which files to use as the reserve and whether to take the file as the reserve or take NOT
                # the file as the reserve...
                selections = {
                    15: [(10, True), (10, False)],  # 15-10 = 5
                    30: [(10, False), (15, False)],
                    50: [(10, False), (15, False), (30, True), (30, False)],
                    100: [(10, False), (15, False), (30, False), (50, False)]
                }
                for supported_reserve, exclude_from_reserve in selections[self.label_ratio]:
                    if supported_reserve == self.reserve_ratio and not exclude_from_reserve:
                        reserve_list_name = 'train' + str(supported_reserve)
                        reserve_list_file = os.path.join(list_root, self.image_list[reserve_list_name])
                        exclude_reserve_list = exclude_from_reserve
                        break
                    elif supported_reserve == self.label_ratio - self.reserve_ratio and exclude_from_reserve:
                        reserve_list_name = 'train' + str(supported_reserve)
                        reserve_list_file = os.path.join(list_root, self.image_list[reserve_list_name])
                        exclude_reserve_list = exclude_from_reserve
                        break

                assert reserve_list_file is not None, "No reserve list file discovered"

                if self.use_reserve:
                    if exclude_reserve_list:
                        exclude_list_file = reserve_list_file
                    else:
                        # Must be in reserve. Overwrite the data list to be the reserve list and set the exclude list
                        # to None so we take the entire thing...
                        exclude_list_file = None
                        data_list_file = reserve_list_file
                else:
                    if exclude_reserve_list:
                        # Must be in reserve. Overwrite the data list to be the reserve list and set the exclude list
                        # to None so we take the entire thing...
                        exclude_list_file = None
                        data_list_file = reserve_list_file
                    else:
                        exclude_list_file = reserve_list_file
            else:
                exclude_list_file = None
        else:
            data_list_file = os.path.join(list_root, self.image_list['test'])
            exclude_list_file = None

        if self.use_train_data:
            full_list_file = os.path.join(list_root, self.image_list['train100'])
            transforms = {'train': TransformTrain()}
        else:
            full_list_file = data_list_file

            if self.single_test_transform:
                transforms = {'test': TransformTest(mean=imagenet_mean, std=imagenet_std)['test9']}
            else:
                transforms = TransformTest(mean=imagenet_mean, std=imagenet_std)

        super().__init__(
            root=files_root,
            samples_per_class=self.samples_per_class,
            class_names=CUB200.CLASSES,
            data_list_file=data_list_file,
            full_list_file=full_list_file,
            exclude_list_file=exclude_list_file,
            use_ssl=use_ssl,
            use_cache=True,
            input_transforms=list(transforms.values()))


if __name__ == "__main__":
    # 10% = 3 samples per class
    # 15% = 5 samples per class
    # 30% = 9 samples per class
    # 50% = 15 samples per class
    # 100% = 30 samples per class

    source_dir_ = 'D:\\Datasets\\cub200'
    ds = CUB200(files_root=source_dir_,
                list_root=os.path.join('..', 'CUB200'),
                is_train=True,
                is_test=False,
                trial_index=0,
                use_ssl=False,
                label_ratio=15,
                reserve_ratio=10,
                use_reserve=True,
                samples_per_class=3)

    print(len(ds))

    ds_loader = DataLoader(ds,
                           num_workers=1,
                           pin_memory=True,
                           shuffle=False,
                           batch_size=50)

    for index_, (img_, class_id_, data_index_, per_class_index_) in enumerate(ds_loader):
        print(index_, data_index_, class_id_, per_class_index_)
        break
