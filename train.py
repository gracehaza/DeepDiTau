import os
import sys
import argparse
import glob
import json
import pickle
import random

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('convertDir', type=str, 
                    help='Directory of input numpy arrays')
parser.add_argument('trainDir', type=str,
                    help='Output directory for trainedresult')
parser.add_argument('--plaid', action='store_true',
                    help='Use plaid (for Mac)')


args = parser.parse_args()

decorrelate = False

# TODO, continue training?
inDir = args.convertDir
outDir = args.trainDir
if os.path.exists(outDir):
    print(outDir,'already exists')
    sys.exit(0)

# if you want to train on Mac, use plaid-ml
usePlaid = args.plaid
if usePlaid:
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

os.makedirs(outDir, exist_ok=True)

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, LSTM, Convolution1D, Masking
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.utils import Sequence

from sklearn.model_selection import train_test_split

import tensorflow as tf
if tf.__version__.startswith('2'):
    from tensorflow.keras import backend as k
else:
    from keras import backend as k
from Disco_tf import mass_decorrelation_loss # mass decorrelation via distance correlation

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

truth_classes = ['lightjet', 'bjet', 'TauHTauH']#, 'TauHTauM', 'TauHTauE']#, 'TauMTauE']

if not usePlaid:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    k.set_session(tf.Session(config=config))

# load all at once
# need to use a generator, but for now, limit to at most 50 GB
max_memory = 50 * 1024 * 1024 * 1024 / len(truth_classes)

nx = 6
ny = 2 if decorrelate else 1
def load_data():
    fnames = {truth: sorted([f for f in glob.glob('{}/output_{}*.x0.npy'.format(inDir,truth)) if 'validation' not in f]) for truth in truth_classes}
    X = {}
    Y = {}
    W = {}
    for truth in truth_classes:
        Xs = []
        Ys = []
        Ws = []
        random.shuffle(fnames[truth])
        for fname in fnames[truth]:
            Xs += [[np.load(fname.replace('.x0.npy','.x{}.npy'.format(i))) for i in range(nx)]]
            Ys += [[np.load(fname.replace('.x0.npy','.y{}.npy'.format(i))) for i in range(ny)]]
            Ws += [np.load(fname.replace('.x0.npy','.w.npy'))]

            #def get_size(*arrays):
            #    return sum([a.size * a.itemsize for a in arrays])

            #total_size = sum([
            #    sum([get_size(*xi) for xi in Xs]),
            #    get_size(*Ys),
            #    get_size(*Ws),
            #])
            #if total_size>max_memory: break

        Ws = [np.reshape(w,(w.shape[0],1)) for w in Ws]

        X[truth] = [np.vstack([Xs[j][i] for j in range(len(Xs))]) for i in range(nx)]
        Y[truth] = [np.vstack([Ys[j][i] for j in range(len(Ys))]) for i in range(ny)]
        W[truth] = np.vstack(Ws) 

        #print(truth)
        #for i in X[truth]:
        #    print(i[:5])
        #print(Y[truth][:5])
        #print(W[truth][:5])

        n = W[truth].shape[0]

        # try dropping rather than weighting them
        #rdrop = np.random.rand(n)
        #keep = W[truth].reshape(n)>rdrop
        #X[truth] = [X[truth][j][keep] for j in range(len(X[truth]))]
        #Y[truth] = Y[truth][keep]
        #W[truth] = W[truth][keep]
        #nn = W[truth].shape[0]
        #W[truth][W[truth].reshape(nn)<1] = 1

    class_counts = [Y[truth][0].shape[0] for truth in truth_classes]
    min_c = min(class_counts)

    #class_weights = [c/sum(class_counts) for c in class_counts]
    #for i,truth in enumerate(truth_classes):
    #    W[truth] = W[truth] * class_weights[i]
    
    X = {truth: [X[truth][i][:min_c] for i in range(nx)] for truth in truth_classes}
    Y = {truth: [Y[truth][i][:min_c] for i in range(ny)] for truth in truth_classes}
    W = {truth: W[truth][:min_c] for truth in truth_classes}

    X = [np.vstack([X[truth][i] for truth in truth_classes]) for i in range(nx)]
    Y = [np.vstack([Y[truth][i] for truth in truth_classes]) for i in range(ny)]
    W = np.vstack([W[truth] for truth in truth_classes])
    W = np.reshape(W,(W.shape[0],))

    args = X + Y + [W]
    res = train_test_split(
        *args,
        shuffle = True,
        test_size = 0.1,
        random_state = 123456,
    )
    X_train = [res[2*i] for i in range(nx)]
    X_test  = [res[2*i+1] for i in range(nx)]
    Y_train = [res[2*nx+2*i] for i in range(ny)]
    Y_test  = [res[2*nx+2*i+1] for i in range(ny)]
    W_train = res[2*(nx+ny)]
    W_test  = res[2*(nx+ny)+1]

    return X_train, X_test, Y_train, Y_test, W_train, W_test


#############
### Model ###
#############

def build_model(input_shapes, num_classes, 
                batchnorm=True, momentum=0.6, 
                dropoutRate=0.2, lr=0.0001, 
                width = 128, depth=4,
                pattern=[], kernel=[],
                doLSTM=True,lstmWidth=128):
    if len(kernel) != len(pattern): kernel = [1]*len(pattern)

    inputs = [Input(shape=s) for s in input_shapes]

    concat = [inputs[0]]

    for i in range(1,len(input_shapes)):
        x = inputs[i]
        for j,p in enumerate(pattern):
            x  = Convolution1D(p, kernel[j], kernel_initializer='lecun_uniform',  activation='relu', name='conv_{}_{}'.format(i,j))(x)
            if j<len(pattern)-1:
                if batchnorm:
                    x = BatchNormalization(momentum=momentum ,name='conv_batchnorm_{}_{}'.format(i,j))(x)
                x = Dropout(dropoutRate,name='conv_dropout_{}_{}'.format(i,j))(x)

        # LSTM
        if doLSTM:
            x = Masking(mask_value=0., name='masking_{}'.format(i))(x)
            x = LSTM(lstmWidth,implementation=2, name='lstm_{}'.format(i))(x)
            if batchnorm:
                x = BatchNormalization(momentum=momentum,name='lstm_batchnorm_{}'.format(i))(x)
            x = Dropout(dropoutRate,name='lstm_dropout_{}'.format(i))(x)
        # flatten
        else:
            x = Flatten()(x)
        concat += [x]



    if len(concat)>1:
        layer = Concatenate()(concat)
    else:
        layer = concat[0]

    for i in range(depth):
        layer = Dense(width, activation='relu', kernel_initializer='lecun_uniform', name='dense_{}'.format(i))(layer)
        if batchnorm:
            layer = BatchNormalization(momentum=momentum, name='dense_batchnorm_{}'.format(i))(layer)
        layer = Dropout(dropoutRate, name='dense_dropout_{}'.format(i))(layer)

    prediction = Dense(num_classes, activation='softmax', kernel_initializer='lecun_uniform', name='ID_pred')(layer)

    if decorrelate:
        mass_input = Input(shape=(1,))
        mass_output = Dense(1, activation='linear',kernel_initializer='normal',name='mass_pred')(mass_input)
        new_prediction = Concatenate()([prediction,mass_output])
        inputs += [mass_input]
        outputs = [new_prediction]
    else:
        outputs = [prediction]

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(lr=lr)
    if decorrelate:
        loss = [mass_decorrelation_loss]
    else:
        loss = ['categorical_crossentropy']
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
    )
    return model

callbacks = [
    ModelCheckpoint('{}/KERAS_check_best_model.h5'.format(outDir), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False),
    EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0, mode='min'),
    CSVLogger('{}/training.csv'.format(outDir)),
]

modelArgs = {
    'doLSTM': True,
    'lstmWidth': 128,
    'depth': 4,
    'width': 128,
    'batchnorm': True,
    'momentum': 0.6, # 0.6-0.85 for large batches (5k+), larger (0.9-0.99) for smaller batches
    'dropoutRate': 0.2,
    'lr': 1e-3,
}

X_train, X_test, Y_train, Y_test, W_train, W_test = load_data()
print([xt.shape for xt in X_train])
print([yt.shape for yt in Y_train])
print(W_train.shape)
nclasses = Y_test[0].shape[1]
print(nclasses)
model = build_model([X_test[i].shape[1:] for i in range(nx)],nclasses,**modelArgs)
model.summary()

if decorrelate:
    _X_train, _Y_train = X_train+[Y_train[1]], np.hstack(Y_train)
else:
    _X_train, _Y_train = X_train, Y_train[0]

print([xi.shape for xi in _X_train])
print(_Y_train.shape)

history = model.fit(_X_train, _Y_train,
                    batch_size = 5000 if decorrelate else 10000, # lower for mass decorrelation
                    epochs = 1000, 
                    verbose = 1,
                    validation_split = 0.1,
                    shuffle = True,
                    sample_weight = W_train,
                    callbacks = callbacks,
                    )

hname = '{}/history.json'.format(outDir)
with open(hname,'w') as f:
    h = history.history
    for k in h:
        h[k] = [float(x) for x in h[k]]
    json.dump(h,f)

# plot loss and accurancy
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epoch_count = range(1, len(loss) + 1)

plt.figure()
plt.plot(epoch_count, loss, 'r--')
plt.plot(epoch_count, val_loss, 'b-')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('{}/loss.png'.format(outDir))

plt.figure()
plt.plot(epoch_count, acc, 'r--')
plt.plot(epoch_count, val_acc, 'b-')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('{}/accuracy.png'.format(outDir))
