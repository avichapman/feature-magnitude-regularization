from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class Task(nn.Module):
    """
    Superclass for any auxiliary task.

    This task should be used as port of each iteration.
     - Get the feature vectors from the encoder.
     - Call `calculate_loss` and add the result to the total loss. Weighting coefficients are handled internally.
     - Call `zero_grad_optimizer` alongside any other calls to optimizers' `zero_grad`.
     - Call the overall loss' `backwards` method.
     - Call `step_optimizer` alongside any other calls to optimizers' `step`.
     - Finally, a call must be made to `end_of_iteration` as well.
    """

    @abstractmethod
    def get_settings_str(self) -> str:
        """:returns a string containing all settings for this task."""
        pass

    # noinspection PyUnusedLocal
    @abstractmethod
    def calculate_loss(self,
                       current_epoch: int,
                       current_iteration: int,
                       feature_vectors: torch.Tensor,
                       classification_logits: torch.Tensor,
                       labels: torch.Tensor,
                       per_class_sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Calculates the loss associated with this task.

        :param current_epoch: The current training epoch.
        :param current_iteration: The current training iteration.
        :param feature_vectors: The output from the upstream encoder.
        :param classification_logits: The output from the classifier.
        :param labels: The associated truth labels.
        :param per_class_sample_indices: The unique index of each sample within their class.
        :return: The loss.
        """
        pass

    @abstractmethod
    def zero_grad_optimizer(self):
        """Call this alongside any other calls to optimizer `zero_grad` in the training iteration."""
        pass

    @abstractmethod
    def step_optimizer(self):
        """Call this alongside any other calls to optimizer `step` in the training iteration."""
        pass

    @abstractmethod
    def end_of_iteration(self):
        """This should be called at the end of each iteration of training, after backpropagation is over."""
        pass

    @abstractmethod
    def end_of_epoch(self, current_epoch: int):
        """
        This should be called at the end of each epoch of training.
        :param current_epoch: The zero-based index of the epoch just finished.
        """
        pass
