import csv
from enum import Enum
from io import StringIO
from typing import List


class FMRLossMethod(Enum):
    """Possible ways to calculate the FMR loss coefficient."""

    PROVIDED = 'provided'
    """
    The coefficient is provided in the configuration.
    """

    STATICALLY_CALCULATED = 'statically_calculated'
    """
    The coefficient is calculated at the start based on dataset size.
    """

    DYNAMICALLY_CALCULATED = 'dynamically_calculated'
    """
    The coefficient is constantly recalculated based on entropy during training.
    """

    META_CALCULATED = 'meta_calculated'
    """
    The coefficient is constantly recalculated based on entropy during training.
    The max coefficient is also dynamically calculated based on the initial entropy.
    """


class FCAMLossRestrictFeatureMode(Enum):
    """
    Possible ways to restrict features with the FCAM loss.
    """

    NO_RESTRICTION = 'none'
    """
    No feature restriction will occur. 
    """

    SELECT_FROM_BOTH = 'selectfromboth'
    """
    The nominated features will be selected from both the prediction and target before inclusion in the loss.
    """

    SUPRESS_OTHERS_ON_TARGET = 'supressothersontarget'
    """
    All non-nominated features on the target will be set to zero.
    """

    INCREASE_ENTROPY = 'increaseentropy'
    """
    Try to maximise entropy across the feature vector.
    """


class Configuration:
    """
    Contains configuration information for the experiment.
    """

    trial_index: int
    label_ratio: int
    samples_per_class: int
    dataset_name: str
    planned_training_iterations: int
    lr: float
    main_classifier_lr_ratio: float
    encoder_model: str
    optimizer_model: str
    scheduler_model: str
    batch_size: int
    is_encoder_pretrained: bool
    pretraining_source: str
    use_fmr: bool
    fmr_loss_coef: float
    fmr_loss_coef_method: FMRLossMethod
    fmr_target_layer: str
    desired_initial_coefficient: float
    max_fmr_coefficient: float
    softmax_tau: float
    technique: str


class Configurator:
    """Loads configurations from a file."""

    @staticmethod
    def load(config_file_path: str) -> List[Configuration]:
        csv_text = ''
        with open(config_file_path, 'r') as in_file:
            for line in in_file:
                if line.find('# ') > -1:
                    continue
                csv_text += line

        configurations = []
        field_names = None
        with StringIO(csv_text) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if field_names is None:
                    field_names = row
                else:
                    field_values = row
                    fields = {name.strip(): value.strip() for name, value in zip(field_names, field_values)}

                    configuration = Configuration()
                    configuration.trial_index = int(fields['TrialIndex'])

                    if 'Dataset' in fields:
                        configuration.dataset_name = fields['Dataset']
                    else:
                        configuration.dataset_name = 'CUB200'

                    if 'LabelRatio' in fields:
                        configuration.label_ratio = int(fields['LabelRatio'])
                    else:
                        configuration.label_ratio = 100

                    if 'SampleCount' in fields:
                        configuration.samples_per_class = int(fields['SampleCount'])
                    else:
                        configuration.samples_per_class = None

                    if 'IterationCount' in fields:
                        configuration.planned_training_iterations = int(fields['IterationCount'])
                    else:
                        configuration.planned_training_iterations = 30000

                    if 'LR' in fields:
                        configuration.lr = float(fields['LR'])
                    else:
                        configuration.lr = 0.001

                    if 'ClsRatio' in fields:
                        configuration.main_classifier_lr_ratio = float(fields['ClsRatio'])
                    else:
                        configuration.main_classifier_lr_ratio = 10.0

                    if 'PretrainedEncoder' in fields:
                        configuration.is_encoder_pretrained = fields['PretrainedEncoder'].upper() == "TRUE"
                    else:
                        configuration.is_encoder_pretrained = False

                    if 'PretrainedSource' in fields:
                        configuration.pretraining_source = fields['PretrainedSource'].lower().strip()
                    else:
                        configuration.pretraining_source = 'imagenet'

                    if 'EncoderModel' in fields:
                        configuration.encoder_model = fields['EncoderModel'].lower()
                    else:
                        configuration.encoder_model = None

                    if 'OptimizerModel' in fields:
                        configuration.optimizer_model = fields['OptimizerModel'].lower()
                    else:
                        configuration.optimizer_model = None

                    if 'SchedulerModel' in fields:
                        configuration.scheduler_model = fields['SchedulerModel'].lower()
                    else:
                        configuration.scheduler_model = 'plateau'

                    if 'BatchSize' in fields:
                        configuration.batch_size = int(fields['BatchSize'])
                    else:
                        configuration.batch_size = None

                    if 'UseFMR' in fields:
                        configuration.use_fmr = fields['UseFMR'].upper() == "TRUE"
                    else:
                        configuration.use_fmr = False

                    if 'FMRCoef' in fields:
                        configuration.fmr_loss_coef = float(fields['FMRCoef'])
                    else:
                        configuration.fmr_loss_coef = None

                    if 'MetaDesiredCoef' in fields:
                        configuration.desired_initial_coefficient = float(fields['MetaDesiredCoef'])
                    else:
                        configuration.desired_initial_coefficient = 50.

                    if 'FMRCoefMethod' in fields:
                        value = fields['FMRCoefMethod'].strip().lower()
                        if value == 'provided':
                            configuration.fmr_loss_coef_method = FMRLossMethod.PROVIDED
                        elif value == 'statically_calculated':
                            configuration.fmr_loss_coef_method = FMRLossMethod.STATICALLY_CALCULATED
                        elif value == 'dynamically_calculated':
                            configuration.fmr_loss_coef_method = FMRLossMethod.DYNAMICALLY_CALCULATED
                        elif value == 'meta_calculated':
                            configuration.fmr_loss_coef_method = FMRLossMethod.META_CALCULATED
                        else:
                            options = ", ".join([str(value.value) for value in FMRLossMethod])
                            raise ValueError(
                                f"Invalid value for 'FMRCoefMethod'. Must be one of {options}")
                    else:
                        configuration.fmr_loss_coef_method = FMRLossMethod.PROVIDED

                    if 'FMRTargetLayer' in fields:
                        configuration.fmr_target_layer = fields['FMRTargetLayer'].strip()
                    else:
                        configuration.fmr_target_layer = 'posterior_features'

                    if 'MaxFMRCoef' in fields:
                        configuration.max_fmr_coefficient = float(fields['MaxFMRCoef'])
                    else:
                        configuration.max_fmr_coefficient = 200.

                    if 'SoftmaxTemp' in fields:
                        configuration.softmax_tau = float(fields['SoftmaxTemp'])
                    else:
                        configuration.softmax_tau = 1.

                    if 'Technique' in fields:
                        configuration.technique = fields['Technique'].strip()
                    else:
                        configuration.technique = 'EDB'

                    configurations.append(configuration)

        return configurations


if __name__ == "__main__":
    print(FCAMLossRestrictFeatureMode['blat'])
    _configurations = Configurator.load('remote/configs.csv')
    print(_configurations)
