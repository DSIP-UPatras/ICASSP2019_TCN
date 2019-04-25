from keras.layers import *
from keras.models import Model, model_from_json
from custom_layers import *
from models import *
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys


def get_model(models_path='.', subject=20):
    filename = ''
    for file in os.listdir(models_path):
        match = re.match(r'O.(.*_{})\.(?:json|h5)'.format(subject), file)
        if match:
            filename = match.group(1)
            json_filename = os.path.join(
                models_path, 'O1' + filename + '.json')
            weights_filename = os.path.join(
                models_path, 'O2' + filename + '.h5')
            break
    print('Found model {},{}'.format(json_filename, weights_filename))
    return json_filename, weights_filename


def attention_model(model):
    for i, layer in enumerate(model.layers):
        if 'attention' in layer.name:
            x_in = layer.input
            x, att = AttentionWithContext(bias=False, return_attention=True, name=layer.name)(x_in)
    y = Dense(53, activation='softmax', name=model.layers[-1].name)(x)
    model.layers.pop()
    model.layers.pop()
    print(x_in, att, y)
    model = Model(model.input, [y, att, x_in])
    return model

RF = int(sys.argv[1])
SUBJECT = int(sys.argv[2])
MODELS_DIR = '../code/models/models_Att{}'.format(RF)
OUT_DIR = '../results/'
model_name = 'TCCNet'
json_filename, weights_filename = get_model(MODELS_DIR, SUBJECT)

# Load model from json
json_file = open(json_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(
    loaded_model_json,
    custom_objects={
        'AttentionWithContext': AttentionWithContext,
        'MaskedConv1D': MaskedConv1D
    }
)

# Make attention_model
model = attention_model(model, att)
model.summary()

# Apply trained weights
model.load_weights(weights_filename, by_name=True)

# Inference
# Load data
X_test, Y_test = np.array(valid_generator.X), valid_generator.y

# Model predict
# generate figures for an input gesture at index 0..len(X_test)
input_index = np.arange(len(X_test))

for index in input_index:
    pred, att, x = model.predict(
        np.expand_dims(
            X_test[index] / np.max(X_test[index]), 0
        )
    )
    y_pred = np.argmax(pred, axis=1)[0]
    y_true = Y_test[index]
    print('True class:', y_true)
    print('Predicted class:', y_pred)
    att = np.squeeze(att)

    # Plot attention
    plt.figure(figsize=(7, 5))
    plt.subplot(311)
    plt.plot(X_test[index] / np.max(X_test[index]))
    plt.ylim(0, 1)
    plt.title('EMG sequence')
    plt.suptitle('True: {}, Prediction: {}'.format(y_true, y_pred))
    plt.subplot(312)
    plt.plot(np.squeeze(x))
    plt.title('Last layer output')
    plt.subplot(313)
    plt.bar(np.arange(len(att)), att, width=1)
    plt.ylim(att.min(), att.max())
    plt.xlabel('time [samples]')
    plt.title('Attention weights')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('{}/{}_{}_{:03}.svg'.format(OUT_DIR,
                                            model_name, subject, index))
    plt.close()
