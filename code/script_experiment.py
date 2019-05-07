
import numpy as np
import tensorflow as tf
import random

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1234)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see:
#    https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0
)
session_conf.gpu_options.allow_growth = True
from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
#   https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

##############################################################################
import sys
import preprocessing
import json
import scipy
import keras
import tensorflow as tf
from keras import optimizers, initializers, regularizers, constraints
from utils import *
from generator import DataGenerator
from models import getNetwork
from sklearn import metrics

print('Keras:', keras.__version__)
print('Tensorflow:', tf.__version__)

# 1. Logging
if len(sys.argv) == 4:
    CONFIG_FILE = str(sys.argv[1])
    SUBJECT = int(sys.argv[2])
    TIMESTAMP = int(sys.argv[3])
else:
    print('Expected different number of arguments. {} were given'.format(len(sys.argv) - 1))
    sys.exit()

with open(CONFIG_FILE) as json_file:
    config_params = json.load(json_file)

LOGGING_FILE_PREFIX = config_params['logging']['log_file'] + '_' + str(TIMESTAMP)
if config_params['logging']['enable']:
    LOGGING_FILE = '../results/L_' + LOGGING_FILE_PREFIX + '.log'
    LOGGING_TENSORBOARD_FILE = '../results/tblogs/L_' + LOGGING_FILE_PREFIX

if config_params['model']['save']:
    MODEL_SAVE_FILE = '../results/models/O1_' + LOGGING_FILE_PREFIX + '_{}.json'
    MODEL_WEIGHTS_SAVE_FILE = '../results/models/O2_' + LOGGING_FILE_PREFIX + '_{}.h5'

METRICS_SAVE_FILE = '../results/metrics/O3_' + LOGGING_FILE_PREFIX + '_{}.mat'


if not os.path.exists(os.path.dirname(METRICS_SAVE_FILE)):
    try:
        os.makedirs(os.path.dirname(METRICS_SAVE_FILE))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

if not os.path.exists(os.path.dirname(MODEL_SAVE_FILE)):
    try:
        os.makedirs(os.path.dirname(MODEL_SAVE_FILE))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
            
if not os.path.exists(os.path.dirname(LOGGING_TENSORBOARD_FILE)):
    try:
        os.makedirs(os.path.dirname(LOGGING_TENSORBOARD_FILE))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
            


print('Logging file: {}'.format(LOGGING_FILE))
print('Tensorboard file: {}'.format(LOGGING_TENSORBOARD_FILE))
print('Model JSON file: {}'.format(MODEL_SAVE_FILE))
print('Model H5 file: {}'.format(MODEL_WEIGHTS_SAVE_FILE))
print('Metrics file: {}'.format(METRICS_SAVE_FILE))

# 2. Config params generator
PARAMS_TRAIN_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = config_params['dataset'].get('train_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_TRAIN_GENERATOR[key] = params_gen[key]

PARAMS_VALID_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = config_params['dataset'].get('valid_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_VALID_GENERATOR[key] = params_gen[key]

# 3. Initialization
INPUT_DIRECTORY = '../dataset/Ninapro-DB1-Proc'
PARAMS_TRAIN_GENERATOR['preprocess_function_1'] = [preprocessing.lpf]
PARAMS_TRAIN_GENERATOR['preprocess_function_1_extra'] = [{'fs': 100}]
PARAMS_TRAIN_GENERATOR['data_type'] = 'rms'
PARAMS_TRAIN_GENERATOR['classes'] = [i for i in range(53)]

PARAMS_VALID_GENERATOR['preprocess_function_1'] = [preprocessing.lpf]
PARAMS_VALID_GENERATOR['preprocess_function_1_extra'] = [{'fs': 100}]
PARAMS_VALID_GENERATOR['data_type'] = 'rms'
PARAMS_VALID_GENERATOR['classes'] = [i for i in range(53)]

SUBJECTS = config_params['dataset'].get('subjects', [i for i in range(1, 28)])
if np.min(SUBJECTS) <= 0 or np.max(SUBJECTS) >= 28:
    raise AssertionError('Subject IDs should be between 1 and 27 inclusive for DB1. Were given {}\n'.format(SUBJECTS))

PARAMS_TRAIN_GENERATOR.pop('input_directory', '')
PARAMS_VALID_GENERATOR.pop('input_directory', '')

MODEL = getNetwork(config_params['model']['name'])

mean_train, mean_test, mean_test_3, mean_test_5 = [], [], [], []
mean_cm = []
mean_train_loss, mean_test_loss = [], []

if config_params['logging']['enable']:
    if os.path.isfile(LOGGING_FILE) == False:
        with open(LOGGING_FILE, 'w') as f:
            f.write(
                'TIMESTAMP: {}\n'
                'KERAS: {}\n'
                'TENSORFLOW: {}\n'
                'DATASET: {}\n'
                'TRAIN_GENERATOR: {}\n'
                'VALID_GENERATOR: {}\n'
                'MODEL: {}\n'
                'MODEL_PARAMS: {}\n'
                'TRAIN_PARAMS: {}\n'.format(
                    TIMESTAMP,
                    keras.__version__, tf.__version__,
                    config_params['dataset']['name'], PARAMS_TRAIN_GENERATOR,
                    PARAMS_VALID_GENERATOR,
                    config_params['model']['name'], config_params['model']['extra'],
                    config_params['training']
                )
            )
            f.write(
                'SUBJECT,TRAIN_SHAPE,TEST_SHAPE,TRAIN_LOSS,TRAIN_ACC,TEST_LOSS,TEST_ACC,TEST_TOP_3_ACC,TEST_TOP_5_ACC\n')

print('Subject: {}'.format(SUBJECT))
input_dir = '{}/subject-{:02d}'.format(INPUT_DIRECTORY, SUBJECT)

train_generator = DataGenerator(input_directory=input_dir, **PARAMS_TRAIN_GENERATOR)
valid_generator = DataGenerator(input_directory=input_dir, **PARAMS_VALID_GENERATOR)
X_test, Y_test, test_reps = valid_generator.get_data()

# print('Train generator:')
# print(train_generator)
# print('Test generator:')
# print(valid_generator)

model = MODEL(
    input_shape=(None, 10),
    classes=train_generator.n_classes,
    **config_params['model']['extra'])
#model.summary()

if config_params['training']['optimizer'] == 'adam':
    optimizer = optimizers.Adam(lr=config_params['training']['l_rate'], epsilon=0.001)
elif config_params['training']['optimizer'] == 'sgd':
    optimizer = optimizers.SGD(lr=config_params['training']['l_rate'], momentum=0.9)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', top_3_accuracy, top_5_accuracy])

train_callbacks = []
if config_params['logging']['enable']:
    tensorboardCallback = MyTensorboard(log_dir=LOGGING_TENSORBOARD_FILE + "/{}".format(SUBJECT),
                                        batch_size=100,
                                        histogram_freq=10)
    train_callbacks.append(tensorboardCallback)
lrScheduler = MyLRScheduler(**config_params['training']['l_rate_schedule'])
train_callbacks.append(lrScheduler)

history = model.fit_generator(train_generator, epochs=config_params['training']['epochs'],
                              validation_data=(X_test,Y_test), callbacks=train_callbacks, verbose=2)
Y_pred = model.predict(X_test)

y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(Y_test, axis=1)

if config_params['model']['save']:
    # serialize model to JSON
    model_json = model.to_json()
    with open(MODEL_SAVE_FILE.format(SUBJECT), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(MODEL_WEIGHTS_SAVE_FILE.format(SUBJECT))
    print("Saved model to disk")


# Confusion Matrix
# C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
cnf_matrix_frame = metrics.confusion_matrix(y_test, y_pred)
if np.array(mean_cm).shape != cnf_matrix_frame.shape:
    mean_cm = cnf_matrix_frame
else:
    mean_cm += cnf_matrix_frame

mean_train.append(history.history['acc'][-1])
mean_test.append(history.history['val_acc'][-1])
mean_train_loss.append(history.history['loss'][-1])
mean_test_loss.append(history.history['val_loss'][-1])
mean_test_3.append(history.history['val_top_3_accuracy'][-1])
mean_test_5.append(history.history['val_top_5_accuracy'][-1])

if config_params['logging']['enable']:
    with open(LOGGING_FILE, 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{}\n'.format(SUBJECT, train_generator.__len__() * PARAMS_TRAIN_GENERATOR['batch_size'], valid_generator.__len__(),
            mean_train_loss[-1], mean_train[-1], mean_test_loss[-1], mean_test[-1], mean_test_3[-1], mean_test_5[-1]))


metrics_dict = {
    'mean_cm': mean_cm,
    'mean_test': mean_test,
    'mean_test_3': mean_test_3,
    'mean_test_5': mean_test_5,
    'mean_train': mean_train,
    'mean_train_loss': mean_train_loss,
    'mean_test_loss': mean_test_loss
}
scipy.io.savemat(METRICS_SAVE_FILE.format(SUBJECT), metrics_dict)
