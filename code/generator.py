# Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import numpy as np
import keras
import scipy.io
import data_augmentation as da
from keras.preprocessing.sequence import pad_sequences


class DataGenerator(keras.utils.Sequence):

    def __init__(self, repetitions, input_directory, batch_size=32, sample_weight=False, dim=(15,10,1),
                 classes=2, shuffle=True, 
                 noise_snr_db=0, scale_sigma=0, window_size=15, window_step=-1, rotation=0, rotation_mask=None, time_warping=0, mag_warping=0, permutation=0,
                 data_type='rms',
                 preprocess_function_1=None, preprocess_function_2=None, preprocess_function_1_extra=None, preprocess_function_2_extra=None,
                 size_factor=1, pad_len=0, pad_value=-10, min_max_norm=True, update_after_epoch=True,
                 label_proc=None, label_proc_extra=None):
        '''
            Initialization
                repetitions -- list, repetition ids to load data from
                input_directory -- str, subject directory to load data from
                batch_size -- int, size of samples to generate
                sample_weight -- bool, whether to calculate sample weights
                dim -- tuple, output shape. The generator yields tensors of size (batch_size, **dim)
                classes -- int or list, which classes to load
                shuffle -- bool, whether to shuffle data or not
                noise_snr_db -- int or list, snr used for generated additive noise (disabled if 0)
                scale_sigma -- float, standard deviation of generated scaling factor (disabled if 0)
                window_size -- int, size of sliding window (disabled if 0)
                window_step -- int, step of sliding windows (disabled if 0)
                rotation -- int, maximum shift for channel rotation (disabled if 0)
                rotation_mask -- list, mask that enables (if 1) or disables (if 0) the rotation of a channel
                time_warping -- float, maximum time warping distance calculated as time_warping_d*len(x)
                mag_warping -- float, standard deviation of magnitude warping
                permutation -- int, maximum number of permutation slices
                data_type -- 'rms' or 'raw', type of data to load
                preprocess_function_1 -- func, function to apply before augmentation
                preprocess_function_2 -- func, function to apply after augmentation
                preprocess_function_1_extra -- dict, extra parameters for preprocessing function 1
                preprocess_function_2_extra -- dict, extra parameters for preprocessing function 2
                size_factor -- int, how many augmentated data are generated
                pad_len -- int, padding for sequences when windowing is disabled
                min_max_norm -- bool, whether to normalize output to [0,1]
        '''

        self.repetitions = repetitions
        self.input_directory = input_directory if isinstance(input_directory, list) else [input_directory]
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.dim = tuple([xi if xi != 0 else None for xi in dim])
        if isinstance(classes, int):
            self.n_classes = classes
            self.classes = [i for i in range(classes)]
        elif isinstance(classes, list):
            self.n_classes = len(classes)
            self.classes = classes
        self.__make_class_index()
        self.n_reps = len(repetitions)
        self.shuffle = shuffle
        self.noise_snr_db = noise_snr_db
        self.scale_sigma = scale_sigma
        self.window_size = window_size
        self.window_step = window_step
        self.rotation = rotation
        self.rotation_mask = rotation_mask
        self.time_warping = time_warping
        self.mag_warping = mag_warping
        self.permutation = permutation
        self.data_type = 'rms' if data_type == 'rms' else 'raw'
        self.preprocess_function_1 = preprocess_function_1
        self.preprocess_function_2 = preprocess_function_2
        self.preprocess_function_1_extra = preprocess_function_1_extra
        self.preprocess_function_2_extra = preprocess_function_2_extra
        self.size_factor = size_factor
        self.pad_len = pad_len
        self.pad_value = float(pad_value)
        self.min_max_norm = min_max_norm
        self.update_after_epoch = update_after_epoch
        self.label_proc = label_proc
        self.label_proc_extra = label_proc_extra
        self.__load_dataset()
        self.__validate_params()
        self.__generate()
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __str__(self):
        return '' + \
            'Classes: {}\n'.format(self.n_classes) + \
            'Class weights: {}\n'.format(self.class_weights) + \
            'Original dataset: {}\n'.format(len(self.X)) + \
            'Augmented dataset: {}\n'.format(len(self.X_aug)) + \
            'Number of sliding windows: {}\n'.format(len(self.x_offsets)) + \
            'Batch size: {}\n'.format(self.batch_size) + \
            'Number of iterations: {}\n'.format(self.__len__()) + \
            'Window size: {}\n'.format(self.window_size) + \
            'Window step: {}\n'.format(self.window_step) + \
            'Pad length: {}\n'.format(self.pad_len) + \
            'Output shape: {}\n'.format(self.dim)

    def __validate_params(self):
        if ((self.dim[0] is None) or (self.pad_len is None)) and ((self.window_size == 0) or (self.window_step == 0)):
            self.dim = (self._max_len, *self.dim[1:])
            self.pad_len = self._max_len
            self.window_step = 0
            self.window_size = 0

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        output = self.__data_generation(indexes)

        return output

    def __generate(self):
        self.__augment()
        self.__make_segments()
        self.indexes = np.arange(len(self.x_offsets))
        if self.batch_size > len(self.x_offsets):
            self.batch_size = len(self.x_offsets)

        self.class_weights = []
        if self.sample_weight:
            self.__make_sample_weights()

        if (self.window_size == 0) and (self.pad_len is not None):
            self.X_aug = pad_sequences(self.X_aug, self.pad_len, dtype=float, value=self.pad_value)

    def on_epoch_end(self):
        '''Applies augmentation and updates indexes after each epoch'''
        if self.update_after_epoch:
            self.__generate()

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        '''Generates data containing batch_size samples'''
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        if self.sample_weight:
            w = np.empty((self.batch_size), dtype=float)

        # Generate data
        for k, index in enumerate(indexes):
            i, j = self.x_offsets[index]
            # Store sample
            if self.window_size != 0:
                x_aug = np.copy(self.X_aug[i][j:j + self.window_size])
            else:
                x_aug = np.copy(self.X_aug[i])

            # Preprocess x
            if self.preprocess_function_2 is not None:
                    if isinstance(self.preprocess_function_2, list):
                        for params, func in zip(self.preprocess_function_2_extra, self.preprocess_function_2):
                            x_aug = func(x_aug, **params)
                    else:
                        x_aug = self.preprocess_function_2(x_aug, **self.preprocess_function_2_extra)

            if self.min_max_norm is True:
                x_aug_masked = np.ma.masked_equal(x_aug, self.pad_value, copy=True)
                max_x = x_aug_masked.max()
                min_x = x_aug_masked.min()
                x_aug_masked = (x_aug_masked - min_x) / (max_x - min_x)
                x_aug = x_aug_masked.data.copy()

            if np.prod(x_aug.shape) == np.prod(self.dim):
                x_aug = np.reshape(x_aug, self.dim)
            else:
                raise Exception('Generated sample dimension mismatch. Found {}, expected {}.'.format(x_aug.shape, self.dim))

            X[k, ] = x_aug

            # Store class
            y[k] = self.class_index[int(self.y_aug[i])]

            if self.sample_weight:
                w[k] = self.class_weights[(y[k])]
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.label_proc is not None:
            y = self.label_proc(y, **self.label_proc_extra)

        output = (X, y)

        if self.sample_weight:
            output += (w,)
        return output

    def __augment(self):
        '''Applies augmentation incrementally'''
        self.X_aug, self.y_aug, self.r_aug = [], [], []
        for i in range(len(self.X)):
            for _ in range(self.size_factor):
                x = np.copy(self.X[i])
                if self.permutation != 0:
                    x = da.permute(x, nPerm=self.permutation)
                if self.rotation != 0:
                    x = da.rotate(x, rotation=self.rotation, mask=self.rotation_mask)
                if self.time_warping != 0:
                    x = da.time_warp(x, sigma=self.time_warping)
                if self.scale_sigma != 0:
                    x = da.scale(x, sigma=self.scale_sigma)
                if self.mag_warping != 0:
                    x = da.mag_warp(x, sigma=self.mag_warping)
                if self.noise_snr_db != 0:
                    x = da.jitter(x, snr_db=self.noise_snr_db)

                if self.permutation or self.rotation or self.time_warping or self.scale_sigma or self.mag_warping or self.noise_snr_db:
                    self.X_aug.append(x)
                    self.y_aug.append(self.y[i])
                    self.r_aug.append(self.r[i])
            self.X_aug.append(self.X[i])
            self.y_aug.append(self.y[i])
            self.r_aug.append(self.r[i])

    def __load_dataset(self):
        '''Loads data and applies preprocess_function_1'''
        X, y, r = [], [], []
        self._max_len = 0
        if 0 in self.classes:
            rest_rep_groups = list(
                zip(
                    np.random.choice(self.repetitions, (self.n_reps), replace=False),
                    np.random.choice([i for i in self.classes if i != 0], (self.n_reps), replace=False)
                )
            )

        for d in range(len(self.input_directory)):
            for label in [i for i in self.classes if i != 0]:
                for rep in self.repetitions:
                    file = '{}/gesture-{:02d}/{}/rep-{:02d}.mat'.format(self.input_directory[d], int(label), self.data_type, int(rep))
                    data = scipy.io.loadmat(file)
                    x = data['emg'].copy()

                    if self.preprocess_function_1 is not None:
                        if isinstance(self.preprocess_function_1, list):
                            for params, func in zip(self.preprocess_function_1_extra, self.preprocess_function_1):
                                x = func(x, **params)
                        else:
                            x = self.preprocess_function_1(x, **self.preprocess_function_1_extra)

                    if len(x) > self._max_len:
                        self._max_len = len(x)
                    X.append(x)
                    y.append(int(np.squeeze(data['stimulus'])[0]))
                    r.append(int(np.squeeze(data['repetition'])[0]))

            if 0 in self.classes:
                for rep, label in rest_rep_groups:
                    file = '{}/gesture-00/{}/rep-{:02d}_{:02d}.mat'.format(self.input_directory[d], self.data_type, int(rep), int(label))
                    data = scipy.io.loadmat(file)
                    x = data['emg'].copy()

                    if self.preprocess_function_1 is not None:
                        if isinstance(self.preprocess_function_1, list):
                            for params, func in zip(self.preprocess_function_1_extra, self.preprocess_function_1):
                                x = func(x, **params)
                        else:
                            x = self.preprocess_function_1(x, **self.preprocess_function_1_extra)

                    if len(x) > self._max_len:
                        self._max_len = len(x)
                    X.append(x)
                    y.append(int(np.squeeze(data['stimulus'])[0]))
                    r.append(int(np.squeeze(data['repetition'])[0]))

        self.X = X
        self.y = y
        self.r = r

    def __make_segments(self):
        '''Creates segments either using predefined step'''
        x_offsets = []

        if self.window_size != 0:
            for i in range(len(self.X_aug)):
                for j in range(0, len(self.X_aug[i]) - self.window_size, self.window_step):
                    x_offsets.append((i, j))
        else:
            x_offsets = [(i, 0) for i in range(len(self.X_aug))]

        self.x_offsets = x_offsets

    def __make_sample_weights(self):
        '''Computes weights for samples'''
        self.class_weights = np.zeros(self.n_classes)
        for index in self.indexes:
            i, j = self.x_offsets[index]
            self.class_weights[self.class_index[int(self.y_aug[i])]] += 1
        self.class_weights = 1 / self.class_weights
        self.class_weights /= np.max(self.class_weights)

    def __make_class_index(self):
        '''Maps class label to 0...len(classes)'''
        self.classes.sort()
        self.class_index = np.zeros(np.max(self.classes) + 1, dtype=int)
        for i, j in enumerate(self.classes):
            self.class_index[j] = i

    def get_data(self):
        '''Retrieves all data of the epoch'''
        X = np.zeros((self.__len__() * self.batch_size, *self.dim))
        y = np.zeros((self.__len__() * self.batch_size, self.n_classes))
        r = np.zeros((self.__len__() * self.batch_size))
        if self.sample_weight:
            w = np.zeros((self.__len__() * self.batch_size))
        for i in range(self.__len__()):
            if self.sample_weight:
                x_, y_, w_ = self.__getitem__(i)
                w[i * self.batch_size:(i + 1) * self.batch_size] = w_
            else:
                x_, y_ = self.__getitem__(i)
            X[i * self.batch_size:(i + 1) * self.batch_size] = x_
            y[i * self.batch_size:(i + 1) * self.batch_size] = y_

        for k, index in enumerate(self.indexes):
            i, j = self.x_offsets[index]
            if k >= len(r):
                break
            r[k] = self.r_aug[i]
        if self.sample_weight:
            return X, y, r, w
        return X, y, r
