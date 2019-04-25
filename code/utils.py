from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import TensorBoard, Callback
import keras.backend as K
import tensorflow as tf
import numpy as np


def top_1_accuracy(y_true, y_pred):
    """
        Calculates top-1 accuracy of the predictions.
        To be used as evaluation metric in model.compile().

        Arguments:
            y_true -- array-like, true labels
            y_pred -- array-like, predicted labels

        Returns:
            top-1 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_3_accuracy(y_true, y_pred):
    """
        Calculates top-1 accuracy of the predictions.
        To be used as evaluation metric in model.compile().

        Arguments:
            y_true -- array-like, true labels
            y_pred -- array-like, predicted labels

        Returns:
            top-1 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_accuracy(y_true, y_pred):
    """
        Calculates top-1 accuracy of the predictions.
        To be used as evaluation metric in model.compile().

        Arguments:
            y_true -- array-like, true labels
            y_pred -- array-like, predicted labels

        Returns:
            top-1 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


class MyTensorboard(TensorBoard):
    """ Tensorboard callback to store the learning rate at the end of each epoch.
    """

    def on_epoch_end(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        lr_summary = tf.Summary(
            value=[tf.Summary.Value(tag='lr', simple_value=lr)])
        self.writer.add_summary(lr_summary, epoch)
        self.writer.flush()
        super(MyTensorboard, self).on_epoch_end(epoch, logs)


class MyLRScheduler(Callback):
    def __init__(self, schedule_type='constant', decay=0, step=0, lr_start=0, lr_end=0, verbose=0):
        super(MyLRScheduler, self).__init__()
        self.schedule_type = schedule_type
        self.decay = float(decay)
        self.step = step
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def schedule(self, epoch):
        """
            Defines the learning rate schedule. This is called at the begin
            of each epoch through the LearningRateScheduler callback.
            Arguments:
                epoch -- integer, current epoch, [0, #epochs-1]

            Returns:
                rate -- calculated learning rate
        """
        if self.schedule_type == 'constant':
            rate = self.lr_start
        elif self.schedule_type == 'decay_once':
            if epoch < self.step:
                rate = self.lr_start
            else:
                rate = self.lr_start * self.decay
        elif self.schedule_type == 'step':
            rate = self.lr_start * (self.decay ** np.floor(epoch / self.step))
        elif self.schedule_type == 'anneal':
            rate = self.lr_start / (1 + self.decay * epoch)
        elif self.schedule_type == 'clr_triangular':
            e = epoch + self.step
            c = np.floor(1 + e / (2 * self.step))
            x = np.abs(e / self.step - 2 * c + 1)
            rate = self.lr_end + (self.lr_start - self.lr_end) * \
                np.maximum(0, (1 - x)) * float(self.decay**(c - 1))
        elif self.schedule_type == 'clr_restarts':
            c = np.floor(epoch / self.step)
            x = 1 + np.cos((epoch % self.step) / self.step * np.pi)
            rate = self.lr_end + 0.5 * \
                (self.lr_start - self.lr_end) * x * self.decay**c
        elif self.schedule_type == 'warmup':
            rate = self.lr_start * \
                np.min(np.pow(epoch, -0.5), epoch * np.pow(self.step, -1.5))
        return float(rate)


DEFAULT_GENERATOR_PARAMS = {
    "repetitions": [],
    "input_directory": '',
    "batch_size": 128,
    "sample_weight": False,
    "dim": [None, ],
    "classes": 5,
    "shuffle": False,
    "noise_snr_db": 0,
    "scale_sigma": 0.,
    "window_size": 0,
    "window_step": 0,
    "rotation": 0,
    "rotation_mask": None,
    "time_warping": 0.,
    "mag_warping": 0.,
    "permutation": 0,
    "data_type": 'rms',
    "preprocess_function_1": None,
    "preprocess_function_2": None,
    "preprocess_function_1_extra": None,
    "preprocess_function_2_extra": None,
    "size_factor": 0,
    "pad_len": None,
    "pad_value": -10,
    "min_max_norm": False,
    "update_after_epoch": False,
    "label_proc": None,
    "label_proc_extra": None
}
