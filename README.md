# feature-magnitude-regularization

This is a Python3 / [Pytorch](https://pytorch.org/) implementation of FMR, as described in the paper **"Enhancing Fine-Grained Visual Recognition in the Low-Data Regime Through Feature Magnitude Regularization"**, by Avraham Chapmanm, Haiming Xiu and Lingqiao Liu.

## Abstract

Training a fine-grained image recognition model with limited data presents a significant challenge, as the subtle differences between categories may not be easily discernible amidst distracting noise patterns. One commonly employed strategy is to leverage pretrained neural networks, which can generate effective feature representations for constructing an image classification model with a restricted dataset. However, these pretrained neural networks are typically trained for different tasks than the fine-grained visual recognition (FGVR) task at hand, which can lead to the extraction of less relevant features. Moreover, in the context of building FGVR models with limited data, these irrelevant features can dominate the training process, overshadowing more useful, generalizable discriminative features.
Our research has identified a surprisingly simple solution to this challenge: we introduce a regularization technique to ensure that the magnitudes of the extracted features are evenly distributed. This regularization is achieved by maximizing the uniformity of feature magnitude distribution, measured through the entropy of the normalized features. The motivation behind this regularization is to remove bias in feature magnitudes from pretrained models, where some features may be more prominent and, consequently, more likely to be used for classification. Additionally, we have developed a dynamic weighting mechanism to adjust the strength of this regularization throughout the learning process. Despite its apparent simplicity, our approach has demonstrated significant performance improvements across various fine-grained visual recognition datasets.

## Setup

To run this code you need to install all packages in the 'requirements.txt' file.
```
pip install -r requirements.txt
```

## Training the model

Use the `train.py` script to train the model. To train the default model on 
CIFAR-10 simply use:

```
python3 feature-magnitude-regularization/train.py --data_dir ./model_data --dataset_list_dir . --dataset_files_dir <DATASETDIR> --configs_path experiments/<CONFIG>.csv --config_index 0 --config_count 1
```

Parameters:
- data_dir: Path to pretrained model backbone weights
- dataset_files_dir: Path to the dataset files
- dataset_list_dir: Path to the dataset lists
- configs_path: Path to the configuration data file.
- config_index: One-based index into the table in the configuration data file.
- config_count: Number of configs to run at once.
- config_skip: Skip the first X configs in the selected range
- use_thorough_test: If provided, test on ten augmentations.
- use_cuda: If 1, attempt to use CUDA
- gpu_id: Specify GPU to run on
- use_relative_dataset_dir: If 1, find appropriate subdir for the selected dataset.
- show_progress_bar: If 1, show progress bar.
- seed: Random seed

## Citation

If you find this code useful please cite us in your work:

```
@inproceedings{Chapman2024FMR,
  title={Enhancing Fine-Grained Visual Recognition in the Low-Data Regime Through Feature Magnitude Regularization},
  author={Avraham Chapman, Haiming Xiu and Lingqiao Liu},
  booktitle={DICTA},
  year={2024}
}
```