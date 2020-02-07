import os
import sys
import argparse
import glob
import json
from array import array
import uproot
import numpy as np

import ROOT
ROOT.gROOT.SetBatch()

parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('convertDir', type=str, 
                    help='Directory of input numpy arrays')
parser.add_argument('trainDir', type=str,
                    help='Directory of trained model')
parser.add_argument('--plaid', action='store_true',
                    help='Use plaid (for Mac)')
parser.add_argument('--decorrelate', action='store_true',
                    help='Use mass decorrelation')

args = parser.parse_args()

inDir = args.convertDir
outDir = args.trainDir
usePlaid = args.plaid
if usePlaid:
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

decorrelate = args.decorrelate

import numpy as np
from keras.models import load_model

import tensorflow as tf
if tf.__version__.startswith('2'):
    from tensorflow.keras import backend as k
else:
    from keras import backend as k
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import unique_labels

from Disco_tf import mass_decorrelation_loss

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

if not usePlaid:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    k.tensorflow_backend.set_session(tf.Session(config=config))


# load all at once
nx = 6
ny = 2 if decorrelate else 1
def load_data():
    fnames = sorted(glob.glob('{}/output_validation_*.x0.npy'.format(inDir)))
    Xs = [[np.load(fname.replace('.x0.npy','.x{}.npy'.format(i))) for i in range(nx)] for fname in fnames]
    Ys = [[np.load(fname.replace('.x0.npy','.y{}.npy'.format(i))) for i in range(ny)] for fname in fnames]

    X = [np.vstack([Xs[j][i] for j in range(len(Xs))]) for i in range(nx)] #if len(Xs[0])>1 else [x[0] for x in Xs]
    Y = [np.vstack([Ys[j][i] for j in range(len(Ys))]) for i in range(ny)]

    rootnames = []
    for fname in fnames:
        with open(fname.replace('_0.x0.npy','.input')) as f:
            rootnames += [line.strip('\n') for line in f.readlines()]

    friendnames = ['{}/{}'.format(outDir,os.path.basename(fname.replace('.x0.npy','.root'))) for fname in fnames]

    if decorrelate:
        X = X+[Y[1]]
        Y = np.hstack(Y)
        return X, Y, rootnames, friendnames
    else:
        Y = Y[0]
        return X, Y, rootnames, friendnames


# TODO load via generator

################
### Plotting ###
################
def plot_confusion_matrix(savename,y_true, y_pred, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(classes)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_ylim(-0.5,cm.shape[0]-0.5)  # hack for matplotlib 3.1.1; 3.1.2 works fine
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('{}.png'.format(savename))

def plot_roc_curve(savename,y_test,y_score,classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    result = {}
    for i in range(len(classes)):
        fpr[i], tpr[i], wp = roc_curve(y_test[:, i], y_score[:, i], drop_intermediate=True)
        roc_auc[i] = auc(fpr[i], tpr[i])
        # overwrite to make smaller lists
        mind = 1e-3
        keep = []
        # via wp
        #prev = 999
        #for wi,w in enumerate(wp):
        #    if abs(prev-w)>mind:
        #        prev = w
        #        keep += [True]
        #    else:
        #        keep += [False]
        # via tpr
        prev = -999
        for ti,t in enumerate(tpr[i]):
            if abs(prev-t)>mind:
                prev = t
                keep += [True]
            else:
                keep += [False]
        keep[0] = True
        keep[-1] = True
        keep = np.array(keep)
        fpr[i] = fpr[i][keep]
        tpr[i] = tpr[i][keep]
        wp = wp[keep]
        class_result = {'tpr': tpr[i].tolist(), 'fpr': fpr[i].tolist(), 'wps':wp.tolist(), 'auc': roc_auc[i]}
        classname = savename+'_'+classes[i]
        result[classes[i]] = class_result
        with open('{}.json'.format(classname),'w') as f:
            json.dump(class_result, f)
    
    for i in range(len(classes)):
        plt.figure()
        lw = 2
        plt.plot(tpr[i], fpr[i], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([1e-4, 1])
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.title(classes[i])
        plt.legend(loc="lower right")
        plt.yscale('log')
        classname = savename+'_'+classes[i]
        plt.savefig('{}.png'.format(classname))

    return result

#############
### Model ###
#############

X, Y, rootnames, friendnames = load_data()
if decorrelate:
    nclasses = Y.shape[1]-1
else:
    nclasses = Y.shape[1]
modelFile = '{}/KERAS_check_best_model.h5'.format(outDir)
# doesnt work with custom loss
#model = load_model(modelFile)
model = load_model(modelFile, custom_objects={'mass_decorrelation_loss':mass_decorrelation_loss})
model.summary()

results = model.evaluate(X, Y,
                    batch_size = 5000, 
                    verbose = 1,
                    )
print('evaluate results: loss, acc =',results)

prediction = model.predict(X,
                    batch_size = 5000,
                    )

print([x.shape for x in X])
print(Y.shape)
print(prediction.shape)

for i in range(nclasses):
    Y_i = Y[Y[:,i].astype(int)==1]
    p_i = prediction[Y[:,i].astype(int)==1]
    print('class',i)
    print('truth')
    print(Y_i[:5])
    print('prediction')
    print(p_i[:5])

np.set_printoptions(precision=2)

Y_pred = np.argmax(prediction[:,:nclasses], axis=1)
Y_truth = np.argmax(Y[:,:nclasses], axis=1)
class_names = ['lightjet', 'bjet', 'TauHTauH']#, 'TauHTauM', 'TauHTauE']#, 'TauMTauE']
plot_confusion_matrix('{}/confusion'.format(outDir), Y_truth, Y_pred, classes=class_names, normalize=True)

# save predictions to root file
with uproot.recreate("{}/prediction.root".format(outDir)) as f:
    f["prediction"] = uproot.newtree(
        {'pred_{}'.format(c): float for i,c in enumerate(class_names)}
    )
    f["prediction"].extend(
        {'pred_{}'.format(c): prediction[:1024,i] for i,c in enumerate(class_names)}
    )

# TODO: validate, written before the training was ready
pvar = 'jet_pt'
etavar = 'jet_eta'
pcol = 0
etacol = 1
pbins = [20,50,100,1000]
npb = len(pbins)-1
etabins = [0,0.5,1.0,1.5,2.0,2.5]
neb = len(etabins)-1

with open('{}/means.json'.format(inDir)) as f:
    result = json.load(f)
means = result['means']
stds = result['stds']
linears = result['linear']
loglinears = result['loglinear']
X0 = X[0]
pvals = X[0][:,pcol]
etavals = X[0][:,etacol]
if pvar in linears:
    pvals = pvals*(linears[pvar][1]-linears[pvar][0])+linears[pvar][0]
elif pvar in loglinears:
    pvals = np.exp(pvals*(np.log(loglinears[pvar][1])-np.log(loglinears[pvar][0]))+np.log(loglinears[pvar][0]))
else:
    pvals = pvals*stds[pvar] + means[pvar]
if etavar in linears:
    etavals = etavals*(linears[etavar][1]-linears[etavar][0])+linears[etavar][0]
else:
    etavals = np.abs(etavals*stds[etavar] + means[etavar])

print(pvals)
print(etavals)

allresults = {'tpr':{},'fpr':{},'wps':{}}

results = plot_roc_curve('{}/roc'.format(outDir),Y,prediction,class_names)
allresults['tpr']['all'] = [results['TauHTauH']['tpr']]
allresults['fpr']['all'] = [results['TauHTauH']['fpr']]
allresults['wps']['all'] = [results['TauHTauH']['wps']]

allresults['tpr']['pbins'] = []
allresults['fpr']['pbins'] = []
allresults['wps']['pbins'] = []
for pb in range(npb):
    thisSel = np.logical_and(pvals>=pbins[pb], pvals<pbins[pb+1])
    thisY = Y[thisSel]
    thisPred = prediction[thisSel]
    results = plot_roc_curve('{}/roc_pBin{}'.format(outDir,pb), thisY, thisPred, class_names)
    allresults['tpr']['pbins'] += [results['TauHTauH']['tpr']]
    allresults['fpr']['pbins'] += [results['TauHTauH']['fpr']]
    allresults['wps']['pbins'] += [results['TauHTauH']['wps']]

allresults['tpr']['etabins'] = []
allresults['fpr']['etabins'] = []
allresults['wps']['etabins'] = []
for eb in range(neb):
    thisSel = np.logical_and(etavals>=etabins[eb], etavals<etabins[eb+1])
    thisY = Y[thisSel]
    thisPred = prediction[thisSel]
    results = plot_roc_curve('{}/roc_etaBin{}'.format(outDir,eb), thisY, thisPred, class_names)
    allresults['tpr']['etabins'] += [results['TauHTauH']['tpr']]
    allresults['fpr']['etabins'] += [results['TauHTauH']['fpr']]
    allresults['wps']['etabins'] += [results['TauHTauH']['wps']]

for pb in range(npb):
    allresults['tpr']['pbin{}'.format(pb)] = []
    allresults['fpr']['pbin{}'.format(pb)] = []
    allresults['wps']['pbin{}'.format(pb)] = []
for eb in range(neb):
    allresults['tpr']['etabin{}'.format(eb)] = []
    allresults['fpr']['etabin{}'.format(eb)] = []
    allresults['wps']['etabin{}'.format(eb)] = []
for pb in range(npb):
    for eb in range(neb):
        thisSel = np.logical_and(np.logical_and(pvals>=pbins[pb], pvals<pbins[pb+1]), np.logical_and(etavals>=etabins[eb], etavals<etabins[eb+1]))
        thisY = Y[thisSel]
        thisPred = prediction[thisSel]
        if thisY.shape[0]:
            results = plot_roc_curve('{}/roc_pBin{}_etaBin{}'.format(outDir,pb,eb), thisY, thisPred, class_names)
        else:
            results = {truth:{'tpr':[],'fpr':[],'wps':[]} for truth in class_names}
        allresults['tpr']['pbin{}'.format(pb)] += [results['TauHTauH']['tpr']]
        allresults['fpr']['pbin{}'.format(pb)] += [results['TauHTauH']['fpr']]
        allresults['wps']['pbin{}'.format(pb)] += [results['TauHTauH']['wps']]
        allresults['tpr']['etabin{}'.format(eb)] += [results['TauHTauH']['tpr']]
        allresults['fpr']['etabin{}'.format(eb)] += [results['TauHTauH']['fpr']]
        allresults['wps']['etabin{}'.format(eb)] += [results['TauHTauH']['wps']]


with open('{}/roc_ditau.json'.format(outDir),'w') as f:
    json.dump(allresults,f)
