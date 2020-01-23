import os
import sys
import argparse
import time
import random
import itertools
import threading
import numpy as np
from multiprocessing import  Pool
import json
import awkward
import uproot
import concurrent.futures
import glob
import h5py
import random
import errno
import logging
from tqdm.auto import tqdm

# this sets the graphics to batch mode
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# define command line arguments
parser = argparse.ArgumentParser(description='Convert')
parser.add_argument('rootDir', type=str, 
                    help='Top level directory of input root files')
parser.add_argument('convertDir', type=str,
                    help='Output directory for converted numpy arrays')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

NTHREADS = 16
parallel = True # TODO: reimplement
entrysteps = 300000
decorrelate = True

# don't over write previous work
# TODO: this can be updated to recover from a cancelled conversion
inDir = args.rootDir
outDir = args.convertDir
if os.path.exists(outDir):
    print(outDir, 'already exists')
    sys.exit(0)

os.makedirs(outDir, exist_ok=True)

# get all root files in the directory
fnames = []
for r, d, f in os.walk(inDir):
    for fname in f:
        if fname.endswith('.root'):
            fnames += [os.path.join(r,fname)]
random.shuffle(fnames)

logging.info('Will convert {} files'.format(len(fnames)))

treename = 'deepntuplizerCA8/tree'

# must create these branches, they are what is output
out_truth = ['lightjet', 'bjet', 'TauHTauH']#, 'TauHTauM', 'TauHTauE']#, 'TauMTauE']
out_vars = ['jet_mass'] # for mass decorrelation
# here make a map for convenience later
truth_map = {
    'lightjet': ['isUD','isS','isG','isC','isCC','isGCC',],
    'bjet': ['isB','isBB','isGBB',],
    'TauHTauH': ['isTauHTauH'],
    #'TauHTauM': ['isTauHTauM'],
    #'TauHTauE': ['isTauHTauE'],
    #'TauMTauE': ['isTauMTauE'],
}

# weight bins to normalize different objects to the same specturm
weight_bins = [
    # jet_pt
    np.array(
        [20,25,30,35,40,50,60,80,100,200,500,1000,2000,7000],
        dtype=float
    ),
    # get_abseta
    np.array(
        [0.0,0.5,1.0,1.5,2.0,2.5],
        dtype=float
    ),
]

# these are the names that you will give the weight bins
weight_bin_labels = ['jet_pt','jet_abseta']
# for plotting, these are the axis labels
weight_bin_axis_labels = [r'Jet $p_{T}$', r'Jet $|\eta|$']
# these are the names of the branches that are read in
weight_branches = ['jet_pt','jet_eta']
# the reference defines the class that all other classes are reweighted to match
reference = 'TauHTauH'
# these are helper branches (perhaps truth info) that are not output to numpy
other_branches = [
]
for t, bs in truth_map.items():
    other_branches += bs

# these are the branches that are output in the numpy arrays
# make sure there is no gen information here, only reco level quantities
# this is just an example, there are many more that can be added
branches = [
    # jet features
    'jet_pt',
    'jet_eta',
    'jet_phi',
    'jet_mass',
    # charged pf features
    'Cpfcan_ptrel',
    'Cpfcan_erel',
    'Cpfcan_phirel',
    'Cpfcan_etarel',
    'Cpfcan_puppiw',
    'Cpfcan_dxy',
    'Cpfcan_dz',
    'Cpfcan_isMu',
    'Cpfcan_isEl',
    # neutral pd features
    'Npfcan_ptrel',
    'Npfcan_erel',
    'Npfcan_puppiw',
    'Npfcan_phirel',
    'Npfcan_etarel',
    'Npfcan_isGamma',
    'Npfcan_HadFrac',
    # secondary vertex features
    'sv_etarel',
    'sv_phirel',
    'sv_mass',
    'sv_ntracks',
    'sv_chi2',
    'sv_ndf',
    'sv_dxy',
    'sv_dxyerr',
    'sv_dxysig',
    'sv_d3d',
    'sv_d3derr',
    'sv_d3dsig',
]

# save the branch names to a text file for use later
with open('{}/branches.txt'.format(outDir),'w') as f:
    for b in branches:
        f.write(b+'\n')

# group the branches based on the number of output numpy blocks
branch_groupings = [
    # jet branches
    [b for b in branches if not any([b.startswith('Cpfcan_'), b.startswith('Npfcan_'), b.startswith('sv_')])],
    # charged pf
    [b for b in branches if b.startswith('Cpfcan_')],
    # neutral pf
    [b for b in branches if b.startswith('Npfcan_')],
    # secondary vertices
    [b for b in branches if b.startswith('sv_')],
]

# how much to zero pad and truncate branches
branch_lengths = {}
branch_lengths.update({b: 10 for b in branches if b.startswith('Cpfcan_')})
branch_lengths.update({b: 10 for b in branches if b.startswith('Npfcan_')})
branch_lengths.update({b: 4 for b in branches if b.startswith('sv_')})

# can optionally linearize branches between [0,1]
linear_branches = {
    'jet_eta': [-3.0,3.0],
    'jet_phi': [-np.pi,np.pi],
}
# or for momentum, log-linearize
loglinear_branches = {
    'jet_pt': [1.5,7000.],
}
# all other branches will be mean normalized to 0 with a sigma of 1

# get weights for the number of each kind of class in the input files
# begin by getting the distributions for each truth class
distributions = {}
with tqdm(unit='files', total=len(fnames), desc='Calculating weights') as pbar:
    for fname in fnames:
        for arrays in uproot.iterate(fname,treename,other_branches+weight_branches,namedecode="utf-8",entrysteps=entrysteps):
            arrays['jet_abseta'] = abs(arrays['jet_eta'])
            keep = np.zeros_like(arrays['jet_pt'], dtype=bool)
            for t, bs in truth_map.items():
                thistruth = np.zeros_like(keep, dtype=bool)
                for b in bs:
                    if b=='isG': # toss 90% of gluon
                        rdf = np.random.rand(*thistruth.shape)
                        thistruth = (thistruth | ((arrays[b]==1) & (rdf<0.1)))
                    else:
                        thistruth = (thistruth | (arrays[b]==1))
                keep = (keep | thistruth)
                arrays[t] = thistruth
    
            for truth in out_truth:
                hist, xedges, yedges = np.histogram2d(
                    arrays[weight_bin_labels[0]][arrays[truth]],
                    arrays[weight_bin_labels[1]][arrays[truth]],
                    weight_bins
                )
                if truth in distributions:
                    distributions[truth] = distributions[truth]+hist
                else:
                    distributions[truth] = hist
        pbar.update(1)

# helper function to divide the truth information
def divide_distributions(a,b):
    out = np.array(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i][j] = a[i][j]/b[i][j] if b[i][j] else 1
    return out

# normalize
for truth in out_truth:
    distributions[truth] = distributions[truth]/distributions[truth].sum()

# now get the weights for each class
weight_distributions = {}
for truth in out_truth:
    weight_distributions[truth] = divide_distributions(distributions[reference],distributions[truth])

# helper function to plot the per class weights
def plot_hist(hist,outname):
    H=hist.T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, H)
    fig.colorbar(im, ax=ax)
    ax.set_xscale("log", nonposx='clip')
    plt.xlabel(weight_bin_axis_labels[0])
    plt.ylabel(weight_bin_axis_labels[1])
    fig.savefig(outname)
    plt.close()

# plot the weights
for truth in out_truth:
    plot_hist(weight_distributions[truth],'{}/weight_{}.png'.format(outDir,truth))

# common transformation for use later
# this will be the first thing called
# for example, you may wish to change a default value to something closer to the bulk distribution
# if a default value is -999 and the main distribution is ~0-1, then you need it change it to something like -1
# so that the network will converge better
def transform(arrays):
    for key in branches:
        if key not in arrays: continue
    return arrays

# get means and sigmas for each branch so that they can be normalized later
means_sum = {key:[] for key in branches}
varis_sum = {key:[] for key in branches}
# dont need to look at them all for means...
subfnames = fnames[:int(0.1*len(fnames))]
with tqdm(unit='files', total=len(subfnames), desc='Calculating means') as pbar:
    for fname in subfnames:
        for arrays in uproot.iterate(fname,treename,branches+other_branches,namedecode="utf-8",entrysteps=entrysteps):
            for key in arrays:
                # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
                if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])
            keep = np.zeros_like(arrays['jet_pt'], dtype=bool)
            for t, bs in truth_map.items():
                thistruth = np.zeros_like(keep, dtype=bool)
                for b in bs:
                    if b=='isG': # toss 90% of gluon
                        rdf = np.random.rand(*thistruth.shape)
                        thistruth = (thistruth | ((arrays[b]==1) & (rdf<0.1)))
                    else:
                        thistruth = (thistruth | (arrays[b]==1))
                keep = (keep | thistruth)
                arrays[t] = thistruth
            for key in arrays:
                arrays[key] = arrays[key][keep]

            arrays = transform(arrays)
            means = {}
            varis = {}
            for key in arrays:
                if key not in branches: continue
                a = arrays[key]
                while isinstance(a,awkward.JaggedArray): a = a.flatten()
                if a.size==0: continue
                means[key] = a[~np.isnan(a)].mean()
                varis[key] = a[~np.isnan(a)].var()
                if not np.isnan(means[key]):
                    # protection against empty arrays. happens with sv for high pt QCD samples...
                    means_sum[key] += [means[key]]
                    varis_sum[key] += [varis[key]]
        pbar.update(1)

means = {key: np.array(means_sum[key]).mean() for key in branches}
varis = {key: np.array(varis_sum[key]).mean() for key in branches}
stds  = {key: np.sqrt(np.array(varis_sum[key]).mean()) for key in branches}

for key in sorted(means):
    logging.info(f'{key}: {means[key]} +/- {stds[key]}')

# we need to save these so that when we evaluate or network later we know what transformation was applied
result = {
    'means':{key:float(item) for key,item in means.items()},
    'stds':{key:float(item) for key,item in stds.items()},
    'linear': {key:item for key,item in linear_branches.items()},
    'loglinear': {key:item for key,item in loglinear_branches.items()},
}
with open('{}/means.json'.format(outDir),'w') as f:
    json.dump(result,f)

# this function calculates the weight that we will apply for a given jet
def weighting(arrays):
    # create abseta
    arrays['jet_abseta'] = abs(arrays['jet_eta'])
    arrays['weight'] = np.zeros(arrays['jet_abseta'].shape)
    for truth in out_truth:
        for xi in range(len(xedges)-1):
            for yi in range(len(yedges)-1):
                mask = ((arrays[truth]) 
                    & (arrays[weight_bin_labels[0]]>xedges[xi]) 
                    & (arrays[weight_bin_labels[0]]<xedges[xi+1])
                    & (arrays[weight_bin_labels[1]]>yedges[yi]) 
                    & (arrays[weight_bin_labels[1]]<yedges[yi+1]))
                arrays['weight'][mask] = weight_distributions[truth][xi][yi]
    return arrays

# this fuction normalizes each feature using the linear, log-linear, or (default) unit normal weighting from above
def normalize(arrays):
    for key in branches:
        if key in linear_branches:
            arrays[key] = (arrays[key].clip(*linear_branches[key])-linear_branches[key][0])/(linear_branches[key][1]-linear_branches[key][0])
        elif key in loglinear_branches:
            arrays[key] = (np.log(arrays[key].clip(*loglinear_branches[key]))-np.log(loglinear_branches[key][0]))/(np.log(loglinear_branches[key][1])-np.log(loglinear_branches[key][0]))
        else:
            arrays[key] = arrays[key]-means[key]
            arrays[key] = arrays[key]/stds[key]
    return arrays

# this function pads and truncates the arrays that can be jagged (like pf candidates)
def padtruncate(arrays):
    for b,l in branch_lengths.items():
        if b not in arrays: continue
        arrays[b] = arrays[b].pad(l,clip=True).fillna(0).regular()
    return arrays

# the main conversion definition
def convert_fname(fname,i):
    for arrays in uproot.iterate(fname,treename,other_branches+branches,namedecode="utf-8",entrysteps=entrysteps):
        # this sets aside 10% for validation checks later
        isval = i%10==1

        # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
        for key in other_branches+branches:
            if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])

        # define truth
        keep = np.zeros_like(arrays['jet_pt'], dtype=bool)
        for t, bs in truth_map.items():
            thistruth = np.zeros_like(keep, dtype=bool)
            for b in bs:
                if b=='isG': # toss 90% of gluon
                    rdf = np.random.rand(*thistruth.shape)
                    thistruth = (thistruth | ((arrays[b]==1) & (rdf<0.1)))
                else:
                    thistruth = (thistruth | (arrays[b]==1))
            keep = (keep | thistruth)
            arrays[t] = thistruth

        # selections
        for key in arrays:
            arrays[key] = arrays[key][keep]

        # calculate weight
        arrays = weighting(arrays)

        # transform
        arrays = transform(arrays)

        # normalize
        arrays = normalize(arrays)

        # zero pad and truncate
        arrays = padtruncate(arrays)
    

        # check for NaN
        for key in arrays:
            a = arrays[key]
            while isinstance(a,awkward.JaggedArray): a = a.flatten()
            n = np.isnan(a)
            while isinstance(n, np.ndarray): n = n.sum()
            if n:
                print(fname)
                print(k)
                print(a[:5])

        def get_output(arrays,out_truth,selection=None):
            if selection is None:
                selection = np.ones_like(arrays['weight'], dtype=bool)
            W = arrays['weight'][selection]
            # note: this stacks the list of arrays that happens if a branch is an array
            X = [np.swapaxes(np.stack([arrays[ab][selection] for ab in groupb]),0,1) for groupb in branch_groupings]
            if decorrelate:
                Y = [np.swapaxes(np.stack([arrays[ot][selection] for ot in out_truth]),0,1),
                     np.swapaxes(np.stack([arrays[ov][selection] for ov in out_vars]),0,1)]
            else:
                Y = np.swapaxes(np.stack([arrays[ot][selection] for ot in out_truth]),0,1)
            return W, X, Y
    
        # convert to numpy
        if isval:
            W, X, Y = get_output(arrays,out_truth)
    
            name = 'output_validation'
            np.save('{}/{}_{}.w.npy'.format(outDir,name,i),W)
            for j,x in enumerate(X):
                np.save('{}/{}_{}.x{}.npy'.format(outDir,name,i,j),x)
            if decorrelate:
                for j,y in enumerate(Y):
                    np.save('{}/{}_{}.y{}.npy'.format(outDir,name,i,j),y)
            else:
                np.save('{}/{}_{}.y.npy'.format(outDir,name,i),Y)
            with open('{}/{}_{}.input'.format(outDir,name,i),'w') as f:
                f.write(fname)
        else:
            W, X, Y = {}, {}, {}
            for truth in out_truth:
                W[truth], X[truth], Y[truth] = get_output(arrays,out_truth,arrays[truth])

            name = 'output'
            for truth in out_truth:
                np.save('{}/{}_{}_{}.w.npy'.format(outDir,name,truth,i),W[truth])
                for j,x in enumerate(X[truth]):
                    np.save('{}/{}_{}_{}.x{}.npy'.format(outDir,name,truth,i,j),x)
                if decorrelate:
                    for j,y in enumerate(Y[truth]):
                        np.save('{}/{}_{}_{}.y{}.npy'.format(outDir,name,truth,i,j),y)
                else:
                    np.save('{}/{}_{}_{}.y.npy'.format(outDir,name,truth,i),Y[truth])
            with open('{}/{}_{}.input'.format(outDir,name,i),'w') as f:
                f.write(fname)

def _futures_handler(futures_set, status=True, unit='items', desc='Processing'):
    try:
        with tqdm(disable=not status, unit=unit, total=len(futures_set), desc=desc) as pbar:
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                futures_set.difference_update(finished)
                while finished:
                    res = finished.pop().result()
                    pbar.update(1)
                time.sleep(0.5)
    except KeyboardInterrupt:
        for job in futures_set:
            job.cancel()
        if status:
            print("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.", file=sys.stderr)
            print("Running jobs:", sum(1 for j in futures_set if j.running()), file=sys.stderr)
    except Exception:
        for job in futures_set:
            job.cancel()
        raise

# iterative fallback
#for i,fname in enumerate(fnames):
#    convert_fname(fname,i)

# TODO: probably not optimal
# currently break apart by file
# would like to break apart by chunks
with concurrent.futures.ThreadPoolExecutor(NTHREADS) as executor:
    futures = set(executor.submit(convert_fname, fname, i) for i, fname in enumerate(fnames))
    _futures_handler(futures, status=True, unit='items', desc='Processing')
