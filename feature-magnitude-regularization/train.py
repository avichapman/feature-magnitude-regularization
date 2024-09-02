import argparse
import asyncio
import os
import random
import statistics
from datetime import datetime
from math import isnan
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.argument_helper import ArgumentHelper
from tools.average_meter import AverageMeter

import numpy as np

from configuration import Configurator, FMRLossMethod
from datasets import load_data, FgvcDataset
from models import FmrModel, FMRTask


class FMRTrainer:
    """
    Trains the Feature-Magnitude-Reduction model.
    """

    def __init__(self,
                 config_name: str,
                 technique: str,
                 trial_index: int,
                 data_list_root: str,
                 data_files_root: str,
                 dataset_name: str,
                 is_encoder_pretrained: bool,
                 pretraining_source: str,
                 use_entropy_increase: bool,
                 entropy_increase_loss_coef: float,
                 entropy_increase_loss_coef_method: FMRLossMethod,
                 entropy_increase_target_layer: str,
                 max_fmr_coefficient: float,
                 softmax_tau: float,
                 model_weights_dir: str,
                 label_ratio: int,
                 samples_per_class: Optional[int],
                 show_progress_bar: bool,
                 desired_initial_coefficient: float = 50.,
                 encoder_model: str = 'resnet18',
                 optimizer_model: str = 'SGD',
                 scheduler_model: str = 'plateau',
                 scheduler_patience: int = 1,
                 test_interval: int = 500,
                 loss_output_interval: int = None,
                 use_thorough_test: bool = False,
                 lr: float = 0.0001,
                 main_classifier_lr_ratio: float = 10.0,
                 planned_training_iterations: int = 30000,
                 batch_size: int = 24,
                 weight_decay: float = 0.0001,
                 num_workers: int = 0,
                 gpu_id: int = 0,
                 lr_scheduler_factor: float = 0.1,
                 seed: int = 0):
        """
        Create a new trainer.
        :param config_name: A unique name for referencing this configuration. The log file and weights are named
        with this string.
        :param technique: The technique to use for fine-tuning. Options are 'EDB' and 'EDS'.
        :param trial_index: We will run more than one trial of each configuration. This is the zero-based trial
         index.
        :param data_list_root: Root directory of dataset lists of files. The data will be in a subdirectory with the
        name supplied in `dataset_name`.
        :param data_files_root: Root directory of the files to load. If the lists contain relative paths, they will be
        relative to this path.
        :param dataset_name: The name of the dataset to load.
        :param use_entropy_increase: If true, use an entropy reduction aux task.
        :param is_encoder_pretrained: If true, the encoder model will start with pre-trained weights.
        :param pretraining_source: The source of the pretrained weights. Options are 'imagenet' and 'moco_v2'.
        :param entropy_increase_loss_coef: The loss coefficient associated with the entropy reduction aux task.
        :param entropy_increase_loss_coef_method: Determines if we calculate the FMR coefficient and how.
        :param max_fmr_coefficient: The maximum FMR coefficient to use in dynamically calculating the FMR Coefficient at
        runtime.
        :param softmax_tau: The temperature value to treat the feature vector with before applying a softmax.
        :param model_weights_dir: The path to the pretrained weights.
        :param label_ratio: Integer from 1-100 describing the percentage of labelled data to serve up.
        :param samples_per_class: If provided, use exactly this many samples per class for training. If not, use
        `label_ratio` as a percentage.
        :param encoder_model: The backbone model to extract features. Can be 'resnet18' or 'resnet50'.
        :param optimizer_model: The model of optimizer to use.
        :param scheduler_model: The scheduler model to use. Options: ['plateau' and 'linear']
        :param scheduler_patience: The scheduler will reduce the base LR if this many evaluations pass without
        improvement.
        :param batch_size: The size of training mini-batches.
        :param test_interval: Run a test after this many iterations of training.
        :param loss_output_interval: Output the losses to the log after this many iterations of training. If not
        provided, defaults to once an epoch.
        :param use_thorough_test: If true, test on ten augmentations.
        :param lr: The initial learning rate for most of hte model.
        :param main_classifier_lr_ratio: The main classifier learning rate will be `lr` x
        `main_classifier_lr_ratio`.
        :param planned_training_iterations: The number of iterations of training to run.
        :param weight_decay: The weight decay value for the optimizer.
        :param show_progress_bar: If true, show a progress bar during training.
        :param num_workers: Number of workers to load data.
        :param gpu_id: The ID of the CUDA device to run the training on.
        :param lr_scheduler_factor: Each time the learning rate scheduler changes the LR, the LR will be multiplied by
        this amount.
        :param seed: Random seed. If zero, one will be chosen randomly and reported in the log for later recreation of
        results.
        """
        self.config_name = ArgumentHelper.check_type(config_name, str)
        self.trial_index = ArgumentHelper.check_type(trial_index, int)
        self.data_list_root = ArgumentHelper.check_type(data_list_root, str)
        self.data_files_root = ArgumentHelper.check_type(data_files_root, str)
        self.dataset_name = ArgumentHelper.check_type(dataset_name, str)
        self.is_encoder_pretrained = ArgumentHelper.check_type(is_encoder_pretrained, bool)
        self.pretraining_source = ArgumentHelper.check_type(pretraining_source, str)
        self.model_weights_dir = ArgumentHelper.check_type(model_weights_dir, str)
        self.label_ratio = ArgumentHelper.check_type(label_ratio, int)
        self.samples_per_class = ArgumentHelper.check_type_or_none(samples_per_class, int)
        self.encoder_model = ArgumentHelper.check_type(encoder_model, str)
        self.optimizer_model = ArgumentHelper.check_type(optimizer_model, str)
        self.scheduler_model = ArgumentHelper.check_type(scheduler_model, str)
        self.scheduler_patience = ArgumentHelper.check_type(scheduler_patience, int)
        self.batch_size = ArgumentHelper.check_type(batch_size, int)
        self.test_interval = ArgumentHelper.check_type(test_interval, int)
        self.loss_output_interval = ArgumentHelper.check_type_or_none(loss_output_interval, int)
        self.use_thorough_test = ArgumentHelper.check_type(use_thorough_test, bool)
        self.num_workers = ArgumentHelper.check_type(num_workers, int)
        self.lr = ArgumentHelper.check_type(lr, float)
        self.main_classifier_lr_ratio = ArgumentHelper.check_type(main_classifier_lr_ratio, float)
        self.planned_training_iterations = ArgumentHelper.check_type(planned_training_iterations, int)
        self.weight_decay = ArgumentHelper.check_type(weight_decay, float)
        self.show_progress_bar = ArgumentHelper.check_type(show_progress_bar, bool)
        self.lr_scheduler_factor = ArgumentHelper.check_type(lr_scheduler_factor, float)
        self.seed = ArgumentHelper.check_type(seed, int)
        self.technique = ArgumentHelper.check_type(technique, str)

        supported_schedulers = ['plateau', 'linear']
        assert self.scheduler_model.lower() in supported_schedulers,\
            f"Schedular '{self.scheduler_model}' is not supported. Must be one of {supported_schedulers}"
        self.use_linear_scheduler = self.scheduler_model.lower() == 'linear'

        supported_techniques = ['EDB', 'EDS']
        assert self.technique in supported_techniques,\
            f"Technique '{self.technique}' is not supported. Must be one of {supported_techniques}"
        self.best_accuracy = 0.0
        self.current_epoch = 0
        self.current_iteration = 0
        self.initial_feature_magnitude = None

        if self.loss_output_interval is None:
            self.loss_output_interval = self.test_interval

        self._set_up_random_seed()
        dataset_loaders, datasets = load_data(
            list_root=self.data_list_root,
            files_root=self.data_files_root,
            dataset_name=self.dataset_name,
            use_ssl=False,
            trial_index=self.trial_index,
            label_ratio=self.label_ratio,
            samples_per_class=self.samples_per_class,
            batch_size=self.batch_size,
            single_test_transform=not self.use_thorough_test,
            num_workers=self.num_workers)
        self.class_count = datasets['train'].class_count
        self.datasets: Dict[str, FgvcDataset] = datasets
        self.dataset_loaders: Dict[str, DataLoader] = dataset_loaders
        self.labeled_data_count = len(self.dataset_loaders["train"])
        self.validation_data_count = len(self.dataset_loaders['val'])
        self.planned_training_epochs = self.planned_training_iterations // self.labeled_data_count

        torch.cuda.set_device(gpu_id)
        self.device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()

        self.state_file_name = "{0}_i{1}.pth.tar".format(config_name, trial_index)
        self.log_file_path = "{0}_i{1}.txt".format(config_name, trial_index)
        self.log = open(self.log_file_path, 'a+')

        self.model = FmrModel(
            pretrained=self.is_encoder_pretrained,
            pretraining_source=self.pretraining_source,
            class_count=self.class_count,
            model_dir=self.model_weights_dir,
            encoder_model=self.encoder_model).to(self.device, non_blocking=True)

        self.fmr_task = FMRTask(
            use_task=ArgumentHelper.check_type(use_entropy_increase, bool),
            target_layer=ArgumentHelper.check_type(entropy_increase_target_layer, str),
            feature_vector_size=self.model.feature_vector_size,
            class_count=self.model.class_count,
            entropy_increase_loss_coef_method=ArgumentHelper.check_type(entropy_increase_loss_coef_method,
                                                                        FMRLossMethod),
            max_fmr_coefficient=ArgumentHelper.check_type(max_fmr_coefficient, float),
            running_average_length=50,
            softmax_tau=ArgumentHelper.check_type(softmax_tau, float),
            desired_initial_coefficient=ArgumentHelper.check_type(desired_initial_coefficient, float),
            coefficient=ArgumentHelper.check_type(entropy_increase_loss_coef, float)
        )

        if self.optimizer_model.upper() == 'SGD':
            self.optimizer = torch.optim.SGD([
                {'params': self.model.get_params_except_classifiers(), 'name': 'model', 'initial_lr': lr},
                {'params': self.model.classifier.parameters(), 'name': 'classifier',
                 'lr': lr * self.main_classifier_lr_ratio,
                 'initial_lr': lr * self.main_classifier_lr_ratio},
            ], lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        elif self.optimizer_model.upper() == 'ADAM':
            self.optimizer = torch.optim.Adam([
                {'params': self.model.get_params_except_classifiers(), 'name': 'model', 'initial_lr': lr},
                {'params': self.model.classifier.parameters(), 'name': 'classifier',
                 'lr': lr * self.main_classifier_lr_ratio,
                 'initial_lr': lr * self.main_classifier_lr_ratio},
            ], lr=self.lr)
        elif self.optimizer_model.upper() == 'ADAMW':
            # The learning rate scales with the batch size. The provided learning rate is assumed to be for a batch
            # size of 256...
            lr = self.lr * self.batch_size / 256.

            self.optimizer = torch.optim.AdamW([
                {'params': self.model.get_params_except_classifiers(), 'name': 'model', 'initial_lr': lr},
                {'params': self.model.classifier.parameters(), 'name': 'classifier',
                 'lr': lr * self.main_classifier_lr_ratio,
                 'initial_lr': lr * self.main_classifier_lr_ratio},
            ], lr=lr, weight_decay=0.01)
        else:
            raise ValueError(f"Unsupported type of optimizer: {self.optimizer_model}")

        self.scheduler_trip_count = 0
        if self.scheduler_model == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=self.lr_scheduler_factor,
                patience=self.scheduler_patience, verbose=False, threshold=1e-4)
        elif self.use_linear_scheduler:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0001, total_iters=self.planned_training_epochs)
        else:
            raise ValueError(f"Unsupported type of scheduler: {self.scheduler_model}")

        self.ce = nn.CrossEntropyLoss()
        self.relu = nn.ReLU(inplace=True)

    def __del__(self):
        if hasattr(self, 'log'):
            self.log.close()

    def run_training(self):
        training_start_time = datetime.now()
        self._write_config_details()

        self.best_accuracy = 0.0
        self.current_iteration = 0

        self.current_epoch = -1  # -1 Means initial statistics run

        labelled_data_iterator = iter(self.dataset_loaders["train"])
        show_live = {
            't_c_ls': True,
            't_e_ls': self.fmr_task.use_task,
            't_entropy': True,
            'f_mag': True,
            'fmr_coef': self.fmr_task.use_dynamic_fmr,
            't_ls': True,
            't_acc': True,
            'e_entropy': True,
            'e_acc': True,
        }

        averages = {
            't_c_ls': AverageMeter(running_average=True, data_length=50),
            't_e_ls': AverageMeter(running_average=True, data_length=50),
            't_entropy': AverageMeter(running_average=True, data_length=50),
            'f_mag': AverageMeter(running_average=True, data_length=50),
            'fmr_coef': AverageMeter(running_average=True, data_length=1),
            't_ls': AverageMeter(running_average=True, data_length=50),
            't_acc': AverageMeter(running_average=True, data_length=50 * self.batch_size),
            'e_entropy': AverageMeter(),
            'e_acc': AverageMeter(),
        }
        self.model.train(True)

        def _create_results_text(live: bool) -> str:
            results_text = 'Iteration {0} (Epoch {1})'.format(self.current_iteration, self.current_epoch)
            for loss_name in averages:
                if not live or show_live[loss_name]:
                    results_text += ', {0}: {loss:.4f}'.format(loss_name, loss=averages[loss_name].get_final_value())
            return results_text

        averages['fmr_coef'].update(self.fmr_task.coefficient)

        if self.show_progress_bar:
            loop = tqdm(range(1, self.planned_training_iterations + 1 + self.labeled_data_count))
        else:
            loop = range(1, self.planned_training_iterations + 1 + self.labeled_data_count)
        for iter_num in loop:
            self.current_iteration = iter_num - self.labeled_data_count

            if self.show_progress_bar:
                loop.set_description(_create_results_text(live=True))
                self._flush_log()

            if self.current_epoch == 0:
                self._write(_create_results_text(live=False), echo=not self.show_progress_bar)
            elif self.current_iteration % self.loss_output_interval == 0 and self.current_epoch >= 0:
                self._write(_create_results_text(live=False), echo=not self.show_progress_bar)
            elif self.current_iteration % self.labeled_data_count == 0 and self.current_epoch >= 0:
                self._write(_create_results_text(live=False), echo=not self.show_progress_bar)

            if self.current_iteration % self.labeled_data_count == 0:
                self.fmr_task.end_of_epoch(self.current_epoch)

                # Re-initialize for a new epoch...
                self.current_epoch += 1
                labelled_data_iterator = iter(self.dataset_loaders["train"])

                # If we just finished the statistical gathering phase, record the initial entropy...
                if self.current_epoch == 0:
                    self.initial_feature_magnitude = averages['f_mag'].get_final_value()
                    self._write(f'Initial Entropy: {self.fmr_task.initial_entropy}')
                    self._write(f'Initial Mean Feature Magnitude: {self.initial_feature_magnitude}')
                    self._write(f'Meta Desired Entropy Coefficient: {self.fmr_task.desired_initial_coefficient}')
                    self._write(
                        f'Meta Calculated Max Entropy Coefficient: {self.fmr_task.calculated_max_fmr_coefficient}')
                else:
                    if self.use_linear_scheduler:
                        self.scheduler.step()

            data_labeled = next(labelled_data_iterator)
            samples = data_labeled[0][0][0].to(self.device, non_blocking=True)
            labels = data_labeled[1].to(self.device, non_blocking=True)
            per_class_sample_indices = data_labeled[3].to(self.device, non_blocking=True)

            self._train_iter(samples, labels, per_class_sample_indices, averages)

            if self.current_iteration % self.test_interval == 1 and self.current_iteration > 1:
                accuracy, entropy = self.evaluate_model()

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self._save_state(best=True)

                self._save_state(best=False)

                if not self.use_linear_scheduler:
                    done_training = self._step_scheduler(accuracy)
                    if done_training:
                        self._write("Run out of patience. Time to stop...")
                        break

                if isnan(accuracy):
                    self._write("Stopping due to NaNing...")
                    break

                averages['e_entropy'].reset()
                averages['e_entropy'].update(entropy)
                averages['e_acc'].reset()
                averages['e_acc'].update(accuracy)

                self._write(_create_results_text(live=False), echo=not self.show_progress_bar)

                # Reset the averages...
                self.model.train(True)

            averages['fmr_coef'].update(self.fmr_task.coefficient)

        self._write("Training Complete.")
        self._write(f"Best Accuracy: {self.best_accuracy}")

        training_duration = datetime.now() - training_start_time
        self._write(f"Total Duration: {training_duration}")

    def evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluates the model's performance and returns its accuracy.
        :returns the accuracy on the evaluation data and the entropy of the feature vectors.
        """
        if self.use_thorough_test:
            return self._evaluate_model_thorough()
        else:
            return self._evaluate_model_fast()

    def _evaluate_model_thorough(self) -> Tuple[float, float]:
        """
        Evaluates the model's performance and returns its accuracy.

        This applies the test on ten different augmentations of the test data.

        :returns the accuracy on the evaluation data and the entropy of the feature vectors.
        """
        with torch.no_grad():
            self.model.eval()

            start_test = True

            entropies = []
            val_len = len(self.dataset_loaders['val'])
            iter_val = iter(self.dataset_loaders['val'])
            for batch_index in range(val_len):
                data = next(iter_val)
                inputs: torch.Tensor = [data[0][j] for j in range(10)]
                labels: torch.Tensor = data[1]

                for j in range(10):
                    inputs[j] = inputs[j].to(self.device)
                labels = labels.to(self.device)

                raw_outputs = []
                for j in range(10):
                    raw_feature_vectors = self.model(inputs[j])
                    raw_logits = self.model.classifier(raw_feature_vectors)
                    raw_outputs.append(raw_logits)

                    entropies.append(self._calculate_entropy(raw_feature_vectors))

                raw_outputs = sum(raw_outputs)
                if start_test:
                    all_raw_outputs = raw_outputs.data.float()
                    all_labels = labels.data.float()
                    start_test = False
                else:
                    all_raw_outputs = torch.cat((all_raw_outputs, raw_outputs.data.float()), 0)
                    all_labels = torch.cat((all_labels, labels.data.float()), 0)

            _, raw_predict = torch.max(all_raw_outputs, 1)

            raw_accuracy = \
                torch.sum(torch.squeeze(raw_predict).float() == all_labels).item() / float(all_labels.size()[0])
            mean_entropy = statistics.mean(entropies)

        return raw_accuracy, mean_entropy

    def _evaluate_model_fast(self) -> Tuple[float, float]:
        """
        Evaluates the model's performance and returns its accuracy.

        This is done with a single augmentation so it can run fast.

        :returns the accuracy on the evaluation data and the entropy of the feature vectors.
        """
        with torch.no_grad():
            self.model.eval()

            start_test = True

            entropies = []
            val_len = len(self.dataset_loaders['val'])
            iter_val = iter(self.dataset_loaders['val'])
            for batch_index in range(val_len):
                data = next(iter_val)
                inputs: torch.Tensor = data[0][0]
                labels: torch.Tensor = data[1]

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                raw_outputs = []
                raw_feature_vectors = self.model(inputs)
                raw_logits = self.model.classifier(raw_feature_vectors)
                raw_outputs.append(raw_logits.detach().cpu())

                entropies.append(self._calculate_entropy(raw_feature_vectors))

                raw_outputs = sum(raw_outputs)
                if start_test:
                    all_raw_outputs = raw_outputs.data.float()
                    all_labels = labels.data.float()
                    start_test = False
                else:
                    all_raw_outputs = torch.cat((all_raw_outputs, raw_outputs.data.float()), 0)
                    all_labels = torch.cat((all_labels, labels.data.float()), 0)

            _, raw_predict = torch.max(all_raw_outputs, 1)

            raw_accuracy = \
                torch.sum(torch.squeeze(raw_predict).float() == all_labels.cpu()).item() / float(all_labels.size()[0])
            mean_entropy = statistics.mean(entropies)

        return raw_accuracy, mean_entropy

    def _train_iter(self,
                    samples: torch.Tensor,
                    labels: torch.Tensor,
                    per_class_sample_indices: torch.Tensor,
                    averages: Dict[str, AverageMeter]):
        raw_feature_vectors = self.model(samples)

        averages['t_entropy'].update(self._calculate_entropy(raw_feature_vectors))
        averages['f_mag'].update(raw_feature_vectors.abs().mean().item())

        logits = self.model.classifier(raw_feature_vectors)

        labelled_mask = labels > -1
        filtered_logits = logits[labelled_mask]
        filtered_labels = labels[labelled_mask]
        filtered_batch_size = filtered_logits.size()[0]

        if filtered_batch_size > 0:
            cls_loss = self.ce(filtered_logits, filtered_labels)
            averages['t_c_ls'].update(cls_loss.item())

            # Calculate the training accuracy...
            _, predictions = torch.max(filtered_logits, 1)
            raw_hit_num = (predictions == filtered_labels).sum().item()
            averages['t_acc'].update(raw_hit_num / float(filtered_batch_size), filtered_batch_size)
        else:
            cls_loss = None

        # Use the logged sum of the vector element exponents to create a loss. This will be lower when the
        # vector has less information...
        loss_entropy_increase = self.fmr_task.calculate_loss(
            self.current_epoch,
            self.current_iteration,
            raw_feature_vectors,
            logits,
            labels,
            per_class_sample_indices)
        if loss_entropy_increase is not None:
            averages['t_e_ls'].update(loss_entropy_increase.item() / self.fmr_task.coefficient)

        total_loss = cls_loss
        if loss_entropy_increase is not None:
            if total_loss is None:
                total_loss = loss_entropy_increase
            else:
                total_loss = total_loss + loss_entropy_increase
        averages['t_ls'].update(total_loss.item())

        # The initial epoch (-1) exists only to start the running averages...
        if self.current_epoch >= 0:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward(retain_graph=True)
            self.optimizer.step()

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

    def _step_scheduler(self, current_accuracy: float) -> bool:
        """
        Steps the scheduler.

        If the scheduler determines that training has stagnated, the learning rates will be reduced or the parameter
        gradients will be calculated.

        If the scheduler has changed settings enough times, we decide that training is over and `True` is returned.
        :param current_accuracy: The evaluation accuracy to use to determine if an LR change is needed.
        :return: True if it is time to stop the training.
        """
        if not self.use_linear_scheduler:
            old_base_lr = self.get_current_base_lr()
            self.scheduler.step(current_accuracy)
            new_base_lr = self.get_current_base_lr()
            if old_base_lr != new_base_lr:
                self._write(f'Reducing LR from {old_base_lr} to {new_base_lr}.', echo=False)
                self.scheduler_trip_count += 1

            # Work out if it is time to stop...
            return self.scheduler_trip_count >= 3

    def get_current_base_lr(self) -> float:
        """
        Returns the base learning rate at this moment.

        This is calculated by the scheduler, which updates the 'prototype_optimizer' periodically.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            # There's only one param group. Grab its LR...
            return param_group['lr']

    def _set_up_random_seed(self):
        """
        Set up seeding. If we provide one, set it. Otherwise, create one randomly and use it in all libraries.
        """
        if self.seed == 0:
            self.seed = random.randint(0, 100000)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def _save_state(self, best: bool = False):
        """
        Saves the current state of training.
        :param best: If true, this is the best score so far. Save it with a separate name.
        """

        state = {
            'config_name': self.config_name,
            'technique': self.technique,
            'batch_size': self.batch_size,
            'test_interval': self.test_interval,
            'encoder_model': self.encoder_model,
            'scheduler_patience': self.scheduler_patience,
            'trial_index': self.trial_index,
            'data_list_root': self.data_list_root,
            'data_files_root': self.data_files_root,
            'dataset_name': self.dataset_name,
            'is_encoder_pretrained': self.is_encoder_pretrained,
            'pretraining_source': self.pretraining_source,
            'use_entropy_increase': self.fmr_task.use_task,
            'entropy_increase_loss_coef': self.fmr_task.coefficient,
            'entropy_increase_loss_coef_method': self.fmr_task.entropy_increase_loss_coef_method,
            'entropy_increase_target_layer': self.fmr_task.target_layer,
            'desired_initial_coefficient': self.fmr_task.desired_initial_coefficient,
            'max_fmr_coefficient': self.fmr_task.max_fmr_coefficient,
            'softmax_tau': self.fmr_task.softmax_tau,
            'label_ratio': self.label_ratio,
            'samples_per_class': self.samples_per_class,
            'num_workers': self.num_workers,
            'lr': self.lr,
            'main_classifier_lr_ratio': self.main_classifier_lr_ratio,
            'planned_training_iterations': self.planned_training_iterations,
            'weight_decay': self.weight_decay,
            'lr_scheduler_factor': self.lr_scheduler_factor,
            'seed': self.seed,
            'model': self.model.state_dict(),
            'best_accuracy': self.best_accuracy,
        }

        if best:
            filename = self.state_file_name.replace('.pth.tar', '_best.pth.tar')
        else:
            filename = self.state_file_name

        torch.save(state, filename)

    def _write_config_details(self):
        self._write('Technique: {0}'.format(self.technique), echo=False)
        self._write('Dataset: {0}'.format(self.dataset_name), echo=False)
        self._write('Label Ratio %: {0}'.format(self.label_ratio), echo=False)
        self._write('Samples Per Class: {0}'.format(self.samples_per_class), echo=False)
        self._write('Training Batch Size: {0}'.format(self.batch_size), echo=False)
        self._write('Test every X Iterations: {0}'.format(self.test_interval), echo=False)
        self._write('Encoder Model: {0}'.format(self.encoder_model), echo=False)
        self._write('Scheduler Patience: {0}'.format(self.scheduler_patience), echo=False)
        self._write('Random Seed: {0}'.format(self.seed), echo=False)
        self._write('Trial Index: {0}'.format(self.trial_index))
        self._write('SGD Weight Decay: {0}'.format(self.weight_decay), echo=False)
        self._write('Worker Count: {0}'.format(self.num_workers), echo=False)
        self._write('Learning Rate: {0}'.format(self.lr), echo=False)
        self._write('Online Classifier LR Ratio: {0}'.format(self.main_classifier_lr_ratio), echo=False)
        self._write('Planned Training Iterations: {0}'.format(self.planned_training_iterations), echo=False)
        self._write('Scheduler LR Decay Factor: {0}'.format(self.lr_scheduler_factor), echo=False)
        self._write('Configuration Name: {0}'.format(self.config_name))
        self._write('Is Encoder Pretrained?: {0}'.format(self.is_encoder_pretrained), echo=False)
        self._write('Encoder Pretrained Weights Source: {0}'.format(self.pretraining_source), echo=False)
        self._write(self.fmr_task.get_settings_str(), echo=False)
        if self.use_thorough_test:
            self._write('Using Test Time Transforms', echo=False)
        else:
            self._write('NOT Using Test Time Transforms', echo=False)

    def _write(self, text: str, echo: bool = True):
        """Write text to the log.

        :argument text The text to write.
        :argument echo If true, write to console as well.
        """
        async def _inline_write():
            if echo:
                print(text)
            self.log.write("{0}\n".format(text))
            self.log.flush()
        asyncio.run(_inline_write())

    def _flush_log(self):
        """Force the log to write to the file."""
        self.log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FMR Experiment')

    parser.add_argument('--data_dir', type=str, default='./data', help='Path to model pretrained weights')
    parser.add_argument('--dataset_files_dir', type=str, help='Path to the dataset files')
    parser.add_argument('--dataset_list_dir', type=str, default='./data', help='Path to the dataset lists')
    parser.add_argument('--configs_path', type=str, help='Path to the configuration data file.')
    parser.add_argument('--config_index', type=int, default=0, help='One-based index into the configuration table.')
    parser.add_argument('--config_count', type=int, default=1, help='Number of configs to run at once.')
    parser.add_argument('--config_skip', type=int, default=0, help='Skip the first X configs in the selected range')
    parser.add_argument('--use_thorough_test', action='store_true', help='If provided, test on ten augmentations.')
    parser.add_argument('--use_cuda', type=int, default=1, help='If 1, attempt to use CUDA')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_relative_dataset_dir', type=int, default=0,
                        help='If 1, find appropriate subdir for the selected dataset.')
    parser.add_argument('--show_progress_bar', type=int, default=0, help='If 1, show progress bar.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    _config_details = Configurator.load(args.configs_path)

    if args.config_index > 0:
        _config_indices = []
        for _i in range(args.config_count):
            _config_indices.append((args.config_index - 1) * args.config_count + _i + 1)

        _subset = []
        for _index in _config_indices:
            _subset.append(_config_details[_index - 1])

        if args.config_skip > 0:
            _subset = _subset[args.config_skip - 1:]
    else:
        _subset = _config_details

    config_index = 0
    for _configuration in _subset:
        config_index += 1

        trial_index_ = _configuration.trial_index
        label_ratio_ = _configuration.label_ratio
        samples_per_class_ = _configuration.samples_per_class
        dataset_name_ = _configuration.dataset_name
        planned_training_iterations_ = _configuration.planned_training_iterations
        lr_ = _configuration.lr
        main_classifier_lr_ratio_ = _configuration.main_classifier_lr_ratio
        encoder_model_ = _configuration.encoder_model
        is_encoder_pretrained_ = _configuration.is_encoder_pretrained
        pretraining_source_ = _configuration.pretraining_source
        use_entropy_increase_ = _configuration.use_entropy_increase
        entropy_increase_loss_coef_ = _configuration.entropy_increase_loss_coef
        entropy_increase_loss_coef_method_ = _configuration.entropy_increase_loss_coef_method
        entropy_increase_target_layer_ = _configuration.entropy_increase_target_layer
        desired_initial_coefficient_ = _configuration.desired_initial_coefficient
        optimizer_model_ = _configuration.optimizer_model
        batch_size_ = _configuration.batch_size
        max_fmr_coefficient_ = _configuration.max_fmr_coefficient
        softmax_tau_ = _configuration.softmax_tau
        technique_ = _configuration.technique
        scheduler_model_ = _configuration.scheduler_model

        if encoder_model_ is None:
            encoder_model_ = 'resnet18'
        if optimizer_model_ is None:
            optimizer_model_ = 'SGD'
        if batch_size_ is None:
            batch_size_ = 24

        _config_subname = '{0}.{1}.{2}.{3}.{4}.{5}.{6}.[{7}].{8}.[{9}]'.format(
            technique_,
            dataset_name_,
            label_ratio_,
            samples_per_class_,
            encoder_model_,
            is_encoder_pretrained_,
            use_entropy_increase_,
            entropy_increase_loss_coef_,
            entropy_increase_loss_coef_method_,
            max_fmr_coefficient_)
        _config_subname += '.{0}.[{1}].[{2}].{3}.{4}.{5}.{6}.[{7}].[{8}].{9}'.format(
            planned_training_iterations_,
            lr_,
            main_classifier_lr_ratio_,
            optimizer_model_,
            scheduler_model_,
            batch_size_,
            pretraining_source_,
            softmax_tau_,
            desired_initial_coefficient_,
            entropy_increase_target_layer_
        )

        if args.use_relative_dataset_dir == 1:
            if dataset_name_.upper() == 'CUB200':
                dataset_files_dir_ = os.path.join(args.dataset_files_dir, 'cub200')
            elif dataset_name_.upper() == 'FGVCAC':
                dataset_files_dir_ = os.path.join(args.dataset_files_dir, 'fgvc')
            elif dataset_name_.upper() == 'CARS':
                dataset_files_dir_ = os.path.join(args.dataset_files_dir, 'StanfordCars')
            elif dataset_name_.upper() == 'INATURALIST':
                dataset_files_dir_ = os.path.join(args.dataset_files_dir, 'iNaturalist')
            else:
                raise ValueError(f"'{dataset_name_}' is not a recognised dataset.")
        else:
            dataset_files_dir_ = args.dataset_files_dir

        print(f'Running Trial Config {config_index} of {len(_subset)}...')

        trainer = FMRTrainer(
            technique=technique_,
            gpu_id=args.gpu_id,
            config_name=_config_subname,
            trial_index=trial_index_,
            data_files_root=dataset_files_dir_,
            data_list_root=args.dataset_list_dir,
            dataset_name=dataset_name_,
            encoder_model=encoder_model_,
            optimizer_model=optimizer_model_,
            scheduler_model=scheduler_model_,
            batch_size=batch_size_,
            is_encoder_pretrained=is_encoder_pretrained_,
            pretraining_source=pretraining_source_,
            use_entropy_increase=use_entropy_increase_,
            entropy_increase_loss_coef=entropy_increase_loss_coef_,
            entropy_increase_loss_coef_method=entropy_increase_loss_coef_method_,
            entropy_increase_target_layer=entropy_increase_target_layer_,
            desired_initial_coefficient=desired_initial_coefficient_,
            max_fmr_coefficient=max_fmr_coefficient_,
            softmax_tau=softmax_tau_,
            model_weights_dir=args.data_dir,
            label_ratio=label_ratio_,
            samples_per_class=samples_per_class_,
            lr=lr_,
            main_classifier_lr_ratio=main_classifier_lr_ratio_,
            planned_training_iterations=planned_training_iterations_,
            show_progress_bar=args.show_progress_bar == 1,
            use_thorough_test=args.use_thorough_test,
            seed=args.seed)
        trainer.run_training()
