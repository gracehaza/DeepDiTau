import os
import sys
import argparse
import time
import random
import itertools
import threading
import numpy as np
from multiprocessing import  Pool, cpu_count
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

NTHREADS = int(cpu_count()*1.5)
parallel = True # note: doesnt actual do anything
entrysteps = 300000
decorrelate = True
import platform
useramdisk = platform.system()=='Linux' # only increases speed a little bit
maxfiles = -1

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
fnames = fnames[:maxfiles]

logging.info('Will convert {} files'.format(len(fnames)))

redirector = 'root://cmseos.fnal.gov/'
def _get_ramdisk(fname):
    if fname.startswith('/dev/shm'): return fname
    fnameComps = fname.split('/')
    if '/store/user' in fname: fnameComps = fnameComps[fnameComps.index('user')+1:]
    if 'gpuscratch' in fname: fnameComps = fnameComps[fnameComps.index('gpuscratch')+1:]
    fname_ramdisk = '/dev/shm/'+str(os.getpid())+'_'.join(fnameComps)
    return fname_ramdisk

def _get_xrootd(fname):
    if fname.startswith('root://'): return fname
    fnameComps = fname.split('/')
    if '/store/user' in fname: fnameComps = fnameComps[fnameComps.index('store'):]
    fname_xrootd = '{}/{}'.format(redirector,'/'.join(fnameComps))
    return fname_xrootd

def _cp_xrootd_to_ramdisk(fname):
    fname_ramdisk = _get_ramdisk(fname)
    fname_xrootd = _get_xrootd(fname)
    if not os.path.exists(fname_ramdisk):
        os.makedirs(os.path.dirname(fname_ramdisk),exist_ok=True)
        os.system('xrdcp {} {} > /dev/null 2>&1'.format(fname_xrootd,fname_ramdisk))

def _rm_ramdisk(fname):
    fname_ramdisk = _get_ramdisk(fname)
    os.system('rm -rf '+fname_ramdisk)

treename = 'deepJetTree/DeepJetTree'

# must create these branches, they are what is output
out_truth = ['lightjet', 'bjet', 'TauHTauH']#, 'TauHTauM', 'TauHTauE']#, 'TauMTauE']
out_vars = ['jet_mass'] # for mass decorrelation
# here make a map for convenience later
truth_map = {
    'lightjet': ['jet_isUD','jet_isS','jet_isG','jet_isC'],
    'bjet': ['jet_isB'],
    'TauHTauH': ['jet_isTauHTauH'],
    #'TauHTauM': ['jet_isTauHTauM'],
    #'TauHTauE': ['jet_isTauHTauE'],
    #'TauMTauE': ['jet_isTauMTauE'],
}

# weight bins to normalize different objects to the same specturm
weight_bins = [
    # jet_pt
    np.array(
        [20,25,30,35,40,50,60,80,100,200,500,1000,2000,7000],
        dtype=float
    ),
    # jet_abseta
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
    'jet_hadronFlavour',
    'jet_partonFlavour',
    "jet_daughter_pdgId",
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
    'jet_jetCharge',
    "jet_chargedMultiplicity",
    "jet_neutralMultiplicity",
    "jet_chargedHadronMultiplicity",
    "jet_neutralHadronMultiplicity",
    "jet_muonMultiplicity",
    "jet_electronMultiplicity",
    "jet_photonMultiplicity",
    "jet_chargedEmEnergy",
    "jet_neutralEmEnergy",
    "jet_chargedHadronEnergy",
    "jet_neutralHadronEnergy",
    "jet_muonEnergy",
    "jet_electronEnergy",
    "jet_photonEnergy",
    "jet_chargedEmEnergyFraction",
    "jet_neutralEmEnergyFraction",
    "jet_chargedHadronEnergyFraction",
    "jet_neutralHadronEnergyFraction",
    "jet_muonEnergyFraction",
    "jet_electronEnergyFraction",
    "jet_photonEnergyFraction",
    "jet_pfJetBProbabilityBJetTags",
    "jet_pfJetProbabilityBJetTags",
    "jet_pfTrackCountingHighEffBJetTags",
    "jet_pfSimpleSecondaryVertexHighEffBJetTags",
    "jet_pfSimpleInclusiveSecondaryVertexHighEffBJetTags",
    "jet_pfCombinedSecondaryVertexV2BJetTags",
    "jet_pfCombinedInclusiveSecondaryVertexV2BJetTags",
    # these have NaNs, need to understand, remove for now
    #"jet_softPFMuonBJetTags",
    #"jet_softPFElectronBJetTags",
    "jet_pfCombinedMVAV2BJetTags",
    "jet_pfCombinedCvsLJetTags",
    "jet_pfCombinedCvsBJetTags",
    "jet_pfDeepCSVJetTags_probb",
    "jet_pfDeepCSVJetTags_probc",
    "jet_pfDeepCSVJetTags_probudsg",
    "jet_pfDeepCSVJetTags_probbb",
    # pf cands
    "jet_daughter_pt",
    "jet_daughter_eta",
    "jet_daughter_phi",
    "jet_daughter_mass",
    "jet_daughter_charge",
    "jet_daughter_etaAtVtx",
    "jet_daughter_phiAtVtx",
    "jet_daughter_vx",
    "jet_daughter_vy",
    "jet_daughter_vz",
    "jet_daughter_dxy",
    # TODO: found inf in dxyError, dzError
    #"jet_daughter_dxyError",
    "jet_daughter_dz",
    #"jet_daughter_dzError",
    "jet_daughter_pixelLayersWithMeasurement",
    "jet_daughter_stripLayersWithMeasurement",
    "jet_daughter_trackerLayersWithMeasurement",
    "jet_daughter_trackHighPurity",
    "jet_daughter_puppiWeight",
    "jet_daughter_puppiWeightNoLep",
    "jet_daughter_isIsolatedChargedHadron",
    "jet_daughter_isStandAloneMuon",
    "jet_daughter_isTrackerMuon",
    "jet_daughter_isGlobalMuon",
    "jet_daughter_isGoodEgamma",
]

charged_hadron_branches = [
    "charged_hadron_pt",
    "charged_hadron_eta",
    "charged_hadron_phi",
    "charged_hadron_charge",
    "charged_hadron_etaAtVtx",
    "charged_hadron_phiAtVtx",
    "charged_hadron_vx",
    "charged_hadron_vy",
    "charged_hadron_vz",
    "charged_hadron_dxy",
    #"charged_hadron_dxyError",
    "charged_hadron_dz",
    #"charged_hadron_dzError",
    "charged_hadron_pixelLayersWithMeasurement",
    "charged_hadron_stripLayersWithMeasurement",
    "charged_hadron_trackerLayersWithMeasurement",
    "charged_hadron_trackHighPurity",
    "charged_hadron_puppiWeight",
    "charged_hadron_puppiWeightNoLep",
    "charged_hadron_isIsolatedChargedHadron",
]

neutral_hadron_branches = [
    "neutral_hadron_pt",
    "neutral_hadron_eta",
    "neutral_hadron_phi",
    "neutral_hadron_puppiWeight",
    "neutral_hadron_puppiWeightNoLep",
]

muon_branches = [
    "muon_pt",
    "muon_eta",
    "muon_phi",
    "muon_charge",
    "muon_etaAtVtx",
    "muon_phiAtVtx",
    "muon_vx",
    "muon_vy",
    "muon_vz",
    "muon_dxy",
    #"muon_dxyError",
    "muon_dz",
    #"muon_dzError",
    "muon_pixelLayersWithMeasurement",
    "muon_stripLayersWithMeasurement",
    "muon_trackerLayersWithMeasurement",
    "muon_trackHighPurity",
    "muon_puppiWeight",
    "muon_isStandAloneMuon",
    "muon_isGlobalMuon",
]

electron_branches = [
    "electron_pt",
    "electron_eta",
    "electron_phi",
    "electron_charge",
    "electron_etaAtVtx",
    "electron_phiAtVtx",
    "electron_vx",
    "electron_vy",
    "electron_vz",
    "electron_dxy",
    #"electron_dxyError",
    "electron_dz",
    #"electron_dzError",
    "electron_pixelLayersWithMeasurement",
    "electron_stripLayersWithMeasurement",
    "electron_trackerLayersWithMeasurement",
    "electron_trackHighPurity",
    "electron_puppiWeight",
]

photon_branches = [
    "photon_pt",
    "photon_eta",
    "photon_phi",
    "photon_puppiWeight",
    "photon_puppiWeightNoLep",
    "photon_isGoodEgamma",
]


# save the branch names to a text file for use later
with open('{}/branches.txt'.format(outDir),'w') as f:
    for b in branches:
        f.write(b+'\n')

# group the branches based on the number of output numpy blocks
branch_groupings = [
    [b for b in branches if not 'jet_daughter_' in b],
    charged_hadron_branches,
    neutral_hadron_branches,
    muon_branches,
    electron_branches,
    photon_branches,
]
all_branches = []
for bg in branch_groupings:
    all_branches += bg

# how much to zero pad and truncate branches
branch_lengths = {}
branch_lengths.update({b:10 for b in charged_hadron_branches})
branch_lengths.update({b:10 for b in neutral_hadron_branches})
branch_lengths.update({b:4 for b in muon_branches})
branch_lengths.update({b:4 for b in electron_branches})
branch_lengths.update({b:4 for b in photon_branches})

# can optionally linearize branches between [0,1]
linear_branches = {
    'jet_eta': [-2.5,2.5],
    'jet_phi': [-np.pi,np.pi],
}
# or for momentum, log-linearize
loglinear_branches = {
    'jet_pt': [20.,7000.],
}

# parallel processing
def _futures_handler(futures_set, status=True, unit='items', desc='Processing'):
    results = []
    try:
        with tqdm(disable=not status, unit=unit, total=len(futures_set), desc=desc) as pbar:
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                futures_set.difference_update(finished)
                while finished:
                    res = finished.pop().result()
                    results += [res]
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
    return results

# all other branches will be mean normalized to 0 with a sigma of 1

def build_truth(arrays,fname=''):
    keep = np.zeros_like(arrays['jet_pt'], dtype=bool)
    for t, bs in truth_map.items():
        thistruth = np.zeros_like(keep, dtype=bool)
        for b in bs:
            # only grab a certain type of jet from each sample
            if 'SUSY' in fname and 'Tau' not in t: continue
            if 'TTJet' in fname and t not in ['bjet','lightjet']: continue
            if 'QCD' in fname and t not in ['bjet','lightjet']: continue
            if t=='lightjet': # toss 90% of lightjet since we have so much
                rdf = np.random.rand(*thistruth.shape)
                thistruth = (thistruth | ((arrays[b]==1) & (rdf<0.1)))
            else:
                thistruth = (thistruth | (arrays[b]==1))
        keep = (keep | thistruth)
        arrays[t] = thistruth
    return arrays, keep
    

# get weights for the number of each kind of class in the input files
# begin by getting the distributions for each truth class
def weight_fname(fname,i):
    distributions = {}
    for arrays in uproot.iterate(fname,treename,other_branches+weight_branches,namedecode="utf-8",entrysteps=entrysteps):
        arrays['jet_abseta'] = abs(arrays['jet_eta'])
        arrays, keep = build_truth(arrays,fname)
    
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
    return distributions, xedges, yedges

with concurrent.futures.ThreadPoolExecutor(NTHREADS) as executor:
    futures = set(executor.submit(weight_fname, fname, i) for i, fname in enumerate(fnames))
    results = _futures_handler(futures, status=True, unit='files', desc='Calculating weights')
    distributions = {}
    for res in results:
        histdist, xedges, yedges = res
        for truth in histdist:
            if truth in distributions:
                distributions[truth] = distributions[truth]+histdist[truth]
            else:
                distributions[truth] = histdist[truth]

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
        # adapt daughters to remove some degrees of freedom
        if key=='jet_daughter_pt':
            arrays[key] = arrays[key]/arrays['jet_pt']
        if key=='jet_daughter_eta':
            arrays[key] = arrays[key]-arrays['jet_eta']
        if key=='jet_daughter_phi':
            arrays[key] = arrays[key]-arrays['jet_phi']
        if key=='jet_daughter_etaAtVtx':
            arrays[key] = arrays[key]-arrays['jet_eta']
        if key=='jet_daughter_phiAtVtx':
            arrays[key] = arrays[key]-arrays['jet_phi']

        # create reduced arrays of daughters
        if 'jet_daughter_' not in key: continue
        chkey = key.replace('jet_daughter_','charged_hadron_')
        nhkey = key.replace('jet_daughter_','neutral_hadron_')
        mkey = key.replace('jet_daughter_','muon_')
        ekey = key.replace('jet_daughter_','electron_')
        pkey = key.replace('jet_daughter_','photon_')
        chmask = ((abs(arrays['jet_daughter_pdgId'])!=13)
                & (abs(arrays['jet_daughter_pdgId'])!=11) 
                & (abs(arrays['jet_daughter_pdgId'])!=22) 
                & (abs(arrays['jet_daughter_charge'])>0))
        nhmask = ((abs(arrays['jet_daughter_pdgId'])!=13)
                & (abs(arrays['jet_daughter_pdgId'])!=11) 
                & (abs(arrays['jet_daughter_pdgId'])!=22) 
                & (abs(arrays['jet_daughter_charge'])==0))
        mmask = (abs(arrays['jet_daughter_pdgId'])==13)
        emask = (abs(arrays['jet_daughter_pdgId'])==11)
        pmask = (abs(arrays['jet_daughter_pdgId'])==22)
        arrays[chkey] = arrays[key][chmask]
        arrays[nhkey] = arrays[key][nhmask]
        arrays[mkey] = arrays[key][mmask]
        arrays[ekey] = arrays[key][emask]
        arrays[pkey] = arrays[key][pmask]
    return arrays

# get means and sigmas for each branch so that they can be normalized later
# dont need to look at them all for means...
def mean_fname(fname,i):
    if useramdisk:
        _cp_xrootd_to_ramdisk(fname)
    means = {key:[] for key in all_branches}
    varis = {key:[] for key in all_branches}
    for arrays in uproot.iterate(fname,treename,branches+other_branches,namedecode="utf-8",entrysteps=entrysteps):
        for key in arrays:
            # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
            if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])
        arrays, keep = build_truth(arrays,fname)
        for key in arrays:
            arrays[key] = arrays[key][keep]

        arrays = transform(arrays)
        for key in arrays:
            if key not in all_branches: continue
            # skip jet_daughter since it was renamed
            if 'jet_daughter_' in key: continue
            a = arrays[key]
            while isinstance(a,awkward.JaggedArray): a = a.flatten()
            if a.size==0: continue
            #a = a[~np.isnan(a)]
            #a = a[~np.isinf(a)]
            m = a.mean()
            v = a.var()
            if np.isnan(m):
                logging.error(f'NaN found: {key}')
                print(fname)
                raise ValueError
            elif np.isinf(m):
                logging.error(f'Inf found: {key}')
                print(fname)
                raise ValueError
            else:
                # protection against empty arrays
                means[key] += [m]
                varis[key] += [v]
    if useramdisk:
        _rm_ramdisk(fname)
    return {'means': means, 'varis': varis}

nfiles = int(0.1*len(fnames))
if nfiles<200:
    nfiles = min([200,len(fnames)])
subfnames = fnames[:nfiles]
with concurrent.futures.ThreadPoolExecutor(NTHREADS) as executor:
    futures = set(executor.submit(mean_fname, fname, i) for i, fname in enumerate(subfnames))
    results = _futures_handler(futures, status=True, unit='files', desc='Calculating means')
    means_sum = {key:[] for key in all_branches}
    varis_sum = {key:[] for key in all_branches}
    for res in results:
        for key in res['means']:
            means_sum[key] += res['means'][key]
            varis_sum[key] += res['varis'][key]

means = {key: np.array(means_sum[key]).mean() for key in all_branches}
varis = {key: np.array(varis_sum[key]).mean() for key in all_branches}
stds  = {key: np.sqrt(np.array(varis_sum[key]).mean()) for key in all_branches}
# protection against divide by 0 later
stds  = {key: std if std else 1.0 for key,std in stds.items()}

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
    for key in all_branches:
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
        arrays[b] = arrays[b].pad(l,clip=True).fillna(0.0).regular()
    return arrays

# the main conversion definition
def convert_fname(fname,i):
    if useramdisk:
        _cp_xrootd_to_ramdisk(fname)
    k = 0 # if there are multiple chunks in the tree
    for arrays in uproot.iterate(fname,treename,other_branches+branches,namedecode="utf-8",entrysteps=entrysteps):
        # this sets aside 10% for validation checks later
        isval = i%10==1

        # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
        for key in other_branches+branches:
            if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])

        # define truth
        arrays, keep = build_truth(arrays,fname)

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
    

        # zero out NaN/Inf
        # we've alread padtruncate into numpy ndarray, so only do the ones we will save
        # and dont worry about jagged arrays
        for key in arrays:
            if key not in all_branches: continue
            a = arrays[key]
            notgood = (np.isnan(a) | np.isinf(a))
            while not isinstance(notgood,np.bool_):
                notgood = notgood.any()
            if notgood:
                a[np.isnan(a)] = 0
                a[np.isinf(a)] = 0
                arrays[key] = a
                

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
                Y = [np.swapaxes(np.stack([arrays[ot][selection] for ot in out_truth]),0,1)]
            return W, X, Y
    
        # convert to numpy
        if isval:
            W, X, Y = get_output(arrays,out_truth)
    
            name = 'output_validation'
            np.save(f'{outDir}/{name}_{i}_{k}.w.npy',W)
            for j,x in enumerate(X):
                np.save(f'{outDir}/{name}_{i}_{k}.x{j}.npy',x)
            for j,y in enumerate(Y):
                np.save(f'{outDir}/{name}_{i}_{k}.y{j}.npy',y)
            with open(f'{outDir}/{name}_{i}.input','w') as f:
                f.write(fname)
        else:
            W, X, Y = {}, {}, {}
            for truth in out_truth:
                W[truth], X[truth], Y[truth] = get_output(arrays,out_truth,arrays[truth])

            name = 'output'
            for truth in out_truth:
                np.save(f'{outDir}/{name}_{truth}_{i}_{k}.w.npy',W[truth])
                for j,x in enumerate(X[truth]):
                    np.save(f'{outDir}/{name}_{truth}_{i}_{k}.x{j}.npy',x)
                for j,y in enumerate(Y[truth]):
                    np.save(f'{outDir}/{name}_{truth}_{i}_{k}.y{j}.npy',y)
            with open(f'{outDir}/{name}_{i}.input','w') as f:
                f.write(fname)

        k += 1
    if useramdisk:
        _rm_ramdisk(fname)

# iterative fallback
#for i,fname in enumerate(fnames):
#    convert_fname(fname,i)

# TODO: probably not optimal
# currently break apart by file
# would like to break apart by chunks
with concurrent.futures.ThreadPoolExecutor(NTHREADS) as executor:
    futures = set(executor.submit(convert_fname, fname, i) for i, fname in enumerate(fnames))
    results = _futures_handler(futures, status=True, unit='files', desc='Converting')
