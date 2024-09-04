import math

import torch
import torch.nn as nn
from typing import Iterator, Optional

from tools.argument_helper import ArgumentHelper
from tools.average_meter import AverageMeter
from configuration import FMRLossMethod
from .task import Task


class FMRTask(Task):
    """
    This is a secondary training task that performs Feature Magnitude Regularisation (FMR).

    It is expected that a feature vector output from an encoder will be fed into this task module. The output will
    be a loss that can be added to the overall training loss.

    This task should be used as port of each iteration.
     - Get the feature vectors from the encoder.
     - Call `calculate_loss` and add the result to the total loss. Weighting coefficients are handled internally.
     - Call `zero_grad_optimizer` alongside any other calls to optimizers' `zero_grad`.
     - Call the overall loss' `backwards` method.
     - Call `step_optimizer` alongside any other calls to optimizers' `step`.
     - Finally, a call must be made to `end_of_iteration` as well.
    """

    def __init__(self,
                 use_task: bool,
                 feature_vector_size: int,
                 class_count: int,
                 fmr_loss_coef_method: FMRLossMethod,
                 max_fmr_coefficient: float,
                 running_average_length: int,
                 target_layer: str = 'posterior_features',
                 softmax_tau: float = 1.0,
                 desired_initial_coefficient: float = 50.,
                 coefficient: float = 10.0):
        """
        Create a new FMR task.

        :param use_task: If true, use this task.
        :param feature_vector_size: The number of expected feature channels output from the encoder that feeds data into
        this task.
        :param class_count: The number of classes in the classifier.
        :param fmr_loss_coef_method: Determines if we calculate the FMR coefficient and how.
        :param max_fmr_coefficient: The maximum FMR coefficient to use in dynamically calculating the FMR Coefficient at
        runtime.
        :param target_layer: Where to apply the entropy loss. Options are 'posterior_features' and 'logits'.
        :param softmax_tau: The temperature value to treat the feature vector with before applying a softmax.
        :param running_average_length: The number of batches' worth of loss to keep in the running average.
        :param desired_initial_coefficient: When using meta FMR coefficient calculation, this is the coefficient we are
        aiming for.
        :param coefficient: The FMR loss coefficient. The loss returned is multiplied by this much.
        """
        super().__init__()

        self.use_task = ArgumentHelper.check_type(use_task, bool)
        self.feature_vector_size = ArgumentHelper.check_type(feature_vector_size, int)
        self.class_count = ArgumentHelper.check_type(class_count, int)
        self.running_average_length = ArgumentHelper.check_type(running_average_length, int)
        self.softmax_tau = ArgumentHelper.check_type(softmax_tau, float)
        self.coefficient = ArgumentHelper.check_type(coefficient, float)
        self.desired_initial_coefficient = ArgumentHelper.check_type(desired_initial_coefficient, float)
        self.fmr_loss_coef_method = ArgumentHelper.check_type(fmr_loss_coef_method, FMRLossMethod)
        self.max_fmr_coefficient = ArgumentHelper.check_type(max_fmr_coefficient, float)

        target_layer_options = ['posterior_features', 'logits']
        self.target_layer = ArgumentHelper.check_type(target_layer, str)
        assert self.target_layer in target_layer_options, f"Invalid target_layer. Options: {target_layer_options}"

        self.use_dynamic_fmr = self.fmr_loss_coef_method in [FMRLossMethod.DYNAMICALLY_CALCULATED,
                                                             FMRLossMethod.META_CALCULATED]

        self.ce = nn.CrossEntropyLoss()
        self.running_entropy_average = AverageMeter(running_average=True, data_length=self.running_average_length)
        self.fmr_adjust_interval = 5

        self.entropy_target: float = self._calculate_entropy_target()
        self.initial_entropy = None
        self.calculated_max_fmr_coefficient = None

        if self.fmr_loss_coef_method == FMRLossMethod.STATICALLY_CALCULATED:
            a = -6.651525319124698e-05
            b = 0.6751367912970034
            self.entropy_increase_loss_coef = math.pow(10., a * len(self.datasets['train']) + b)
            self._write(f'Setting Entropy Increase Loss to {self.entropy_increase_loss_coef}.')

    def get_settings_str(self) -> str:
        """:returns a string containing all settings for this task."""

        msg = 'Use FMR Aux Task?: {0}\n'.format(self.use_task)
        msg += 'FMR Aux Task Loss Coefficient: {0}\n'.format(self.coefficient)
        msg += 'FMR Target Layer: {0}\n'.format(self.target_layer)
        msg += 'FMR Aux Task Loss Coefficient Determination Method: {0}\n'.format(self.fmr_loss_coef_method)
        msg += 'Target Entropy for Dynamic FMR Loss Coefficient: {0}\n'.format(self.entropy_target)
        msg += 'Max Coefficient for Dynamic FMR Loss Coefficient: {0}\n'.format(self.max_fmr_coefficient)
        msg += 'Temperature to treat feature vector before softmax: {0}'.format(self.softmax_tau)
        return msg

    def calculate_loss(self,
                       current_epoch: int,
                       current_iteration: int,
                       feature_vectors: torch.Tensor,
                       classification_logits: torch.Tensor,
                       labels: torch.Tensor,
                       per_class_sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Runs input through the model and returns the prediction label and feature vector.

        :param current_epoch: The current training epoch.
        :param current_iteration: The current training iteration.
        :param feature_vectors: The output from the upstream encoder.
        :param classification_logits: The output from the classifier.
        :param labels: The associated truth labels. These are used for sorting the inputs into the appropriate private
        head.
        :param per_class_sample_indices: The unique index of each sample within their class.
        :return: The loss.
        """
        if self.target_layer == 'logits':
            vectors = classification_logits
        else:
            vectors = feature_vectors

        self.running_entropy_average.update(self._calculate_entropy(vectors))

        should_adjust_coefficient = False
        if current_epoch == 0:
            should_adjust_coefficient = True
        elif current_iteration % self.fmr_adjust_interval == 1:
            should_adjust_coefficient = True

        if should_adjust_coefficient:
            self._adjust_fmr_coefficient(current_epoch)

        if not self.use_task:
            return None
        if current_epoch < 0:
            return None

        # Cross Entropy equation implicitly takes the softmax of the input and assumes the target is already
        # a probability distribution...
        adjusted_vectors = vectors / self.softmax_tau
        probability_distribution = adjusted_vectors.float().softmax(dim=-1)
        loss_entropy_increase = -self.ce(adjusted_vectors, probability_distribution)

        return loss_entropy_increase * self.coefficient

    def zero_grad_optimizer(self):
        """Call this alongside any other calls to optimizer `zero_grad` in the training iteration."""
        pass

    def step_optimizer(self):
        """Call this alongside any other calls to optimizer `step` in the training iteration."""
        pass

    def end_of_iteration(self):
        """This should be called at the end of each iteration of training, after backpropagation is over."""
        pass

    def end_of_epoch(self, current_epoch: int):
        """
        This should be called at the end of each epoch of training.
        :param current_epoch: The zero-based index of the epoch just finished.
        """
        if current_epoch == -1:
            self.initial_entropy = self.running_entropy_average.get_final_value()
            self.calculated_max_fmr_coefficient = (self.desired_initial_coefficient * self.entropy_target) / (
                        self.entropy_target - self.initial_entropy)

    def _adjust_fmr_coefficient(self, current_epoch: int):
        """Changes the FMR coefficient dynamically based on the observed entropy."""
        if self.use_dynamic_fmr and current_epoch >= 0:
            current_entropy = self.running_entropy_average.get_final_value()
            target_entropy = self.entropy_target

            if self.fmr_loss_coef_method == FMRLossMethod.META_CALCULATED:
                max_fmr_coefficient = self.calculated_max_fmr_coefficient
            else:
                max_fmr_coefficient = self.max_fmr_coefficient

            self.coefficient = ((target_entropy - current_entropy) / target_entropy) * max_fmr_coefficient

    def _calculate_entropy_target(self) -> float:
        """
        Calculates the highest possible entropy for a vector produced by the model.
        """

        if self.target_layer == 'logits':
            vector_size = self.class_count
        else:
            vector_size = self.feature_vector_size
        sample_vector = torch.ones((1, vector_size)) / vector_size  # Maximum entropy vector
        return self._calculate_entropy(sample_vector)

    def _create_optimizer(self, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.SGD:
        return torch.optim.SGD(parameters,
                               lr=self.lr,
                               momentum=self.sgd_momentum,
                               weight_decay=self.sgd_decay)

    @staticmethod
    def _calculate_entropy(feature_vectors: torch.Tensor) -> float:
        """
        Passes a batch of feature vectors through a softmax and then calculates the entropy of the resulting probability
        distributions.
        :param feature_vectors: The feature vectors to treat.
        :return: The average entropy of the batch.
        """
        batch_size = feature_vectors.size()[0]
        feature_count = feature_vectors.size()[1]
        probability_distributions = feature_vectors.softmax(dim=-1)

        entropies = torch.zeros((batch_size,), device=feature_vectors.device)

        # Compute entropy
        for i in range(feature_count):
            entry_batch = probability_distributions[:, i]
            entropies = entropies - entry_batch * entry_batch.log()

        return entropies.mean().item()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
