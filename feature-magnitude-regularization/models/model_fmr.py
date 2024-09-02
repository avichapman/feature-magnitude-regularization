import os
from typing import Iterator

import torch.nn as nn
from torch.utils import model_zoo
import torch.utils.data

from ..tools.argument_helper import ArgumentHelper
from model_resnet import resnet18, resnet50

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class FmrModel(nn.Module):
    """
    Model for performing feature magnitude regularization.
    """

    def __init__(self,
                 class_count: int,
                 model_dir: str,
                 encoder_model: str = 'resnet18',
                 pretrained: bool = True,
                 pretraining_source: str = 'imagenet'):
        """
        Creates a new model.
        :param class_count: The number of classes to classify.
        :param model_dir: The directory where model data is stored.
        :param encoder_model: The pretrained model to fine-tune.
        :param pretrained: If true, load pretrained weights.
        :param pretraining_source: The source of pretrained weights.
        """
        super().__init__()

        self.model_dir = ArgumentHelper.check_type(model_dir, str)
        self.encoder_model = ArgumentHelper.check_type(encoder_model, str).lower()
        self.class_count = ArgumentHelper.check_type(class_count, int)
        self.pretraining_source = ArgumentHelper.check_type(pretraining_source, str).lower()

        pretraining_source_options = ['imagenet', 'moco_v2', 'dino']
        if self.pretraining_source not in pretraining_source_options:
            raise ValueError(f"Illegal option for pretrained weights source ('{pretraining_source}'). " +
                             f"Options are: {pretraining_source_options}")

        # See if there's a local copy of the pretrained weights...
        if self.pretraining_source == 'imagenet':
            local_weights_file = os.path.abspath(os.path.join(model_dir, f'{encoder_model}_imagenet.pth'))
            if not os.path.exists(local_weights_file):
                local_weights_file = os.path.abspath(os.path.join(model_dir, f'{encoder_model}.pth'))
                if not os.path.exists(local_weights_file):
                    local_weights_file = None

        elif self.pretraining_source == 'dino':
            local_weights_file = os.path.abspath(os.path.join(model_dir, f'{encoder_model}_dino.pth'))
            if not os.path.exists(local_weights_file):
                local_weights_file = None
        else:
            local_weights_file = os.path.abspath(os.path.join(model_dir, f'{encoder_model}_moco_v2.pth.tar'))
            if not os.path.exists(local_weights_file):
                local_weights_file = None

        if encoder_model == 'resnet18':
            self.encoder = resnet18(pretrained=False)  # Loaded below
            self.feature_vector_size = 512
            self.features_per_layer = [64, 128, 256, 512]
        elif encoder_model == 'resnet50':
            self.encoder = resnet50(pretrained=False)  # Loaded below
            self.feature_vector_size = 2048
            self.features_per_layer = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"'{encoder_model}' is not a supported encoder model.")

        self.feature_count = self.encoder.out_features

        # Load pretrained weights before expanding the network...
        if pretrained:
            print('Loading Pretrained Weights from', local_weights_file)
            if local_weights_file is not None:
                weights = model_zoo.load_url(local_weights_file, model_dir=model_dir)

                if self.pretraining_source == 'moco_v2':
                    # The weights have different names from moco_v2, change them to be compatible...
                    new_weights = {}
                    for key in weights['state_dict']:
                        if key.startswith('module.encoder_q.'):
                            new_key = key.replace('module.encoder_q.', '')
                            new_weights[new_key] = weights['state_dict'][key]
                    weights = new_weights

                load_result = self.encoder.load_state_dict(weights, strict=False)
            else:
                print('Couldn\'t find it. Trying online...')
                load_result = self.encoder.load_state_dict(
                    model_zoo.load_url(model_urls[encoder_model], model_dir=model_dir), strict=False)
            print('...Done. Result:', load_result)

        self.classifier = nn.Linear(in_features=self.feature_count,
                                    out_features=self.class_count)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def get_blank_encoder(self) -> nn.Module:
        """
        Returns an uninitialised encoder network that matches this model's network.
        """
        if self.encoder_model == 'resnet18':
            return resnet18(pretrained=False)
        elif self.encoder_model == 'resnet50':
            return resnet50(pretrained=False)
        else:
            raise ValueError(f"'{self.encoder_model}' is not a supported encoder model.")

    def get_params_except_classifiers(self) -> Iterator[torch.nn.Parameter]:
        """Gets the parameters for training the backbone only."""
        all_params = {name: param for name, param in self.named_parameters()}
        non_classifier_params = [all_params[name] for name in all_params if name.find("classifier.") == -1]
        return non_classifier_params

    def get_activation_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs data through the model and produces a feature activation map.
        :param x: The data to run through the model.
        :return: The model features as a spatial map.
        """
        _, feature_map = self.encoder(x)

        return feature_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs data through the model and produces both CAMs and feature vectors.
        :param x: The data to run through the model.
        :return: The model features
        """
        output = self.encoder(x)

        return output[0]
