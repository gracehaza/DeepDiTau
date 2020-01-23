#!/usr/bin/env python
import os
import json
import subprocess
from utilities import load, dump, get_das
import logging


update = True # requery everything
verbose = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# NanoAODv6
nano_tag = 'Nano25Oct2019'


# mc
# note: this doesnt get the "new_pmx" or weirdly named "ext" samples... maybe okay?
# for example: RunIIFall17NanoAODv6-PU2017_12Apr2018_Nano25Oct2019_ext_102X_mc2017_realistic_v7
#              RunIIFall17NanoAODv6-PU2017_12Apr2018_Nano25Oct2019_new_pmx_102X_mc2017_realistic_v7
year_tags = {
    '2016': f'RunIISummer16NanoAODv6-PUMoriond17_{nano_tag}_102X_mcRun2_asymptotic_v7',
    '2017': f'RunIIFall17NanoAODv6-PU2017_12Apr2018_{nano_tag}_102X_mc2017_realistic_v7',
    '2018': f'RunIIAutumn18NanoAODv6-{nano_tag}_102X_upgrade2018_realistic_v20',
}

# datasets (note, tune changes between 2016 and 2017/2018, but not always)
datasets = [
    # TT
    'TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8',
    'TTJets_TuneCUETP8M2T4_13TeV-madgraphMLM-pythia8',
    'TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8',
    'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8',
    # HAA
    'SUSY*HToAA*AToMuMu*AToTauTau*',
    # QCD
    # note: will also match patterns like Pt-*to*_, so manually delete those (muon enriched, for example)
    'QCD_Pt_*to*_TuneCUETP8M1_13TeV_pythia8',
    'QCD_Pt_*to*_TuneCP5_13TeV_pythia8',
]


def get_mc(update=False,verbose=False):


    for year in year_tags:
        fname = f'mc_{year}'
        result = load(fname)
        for dataset in datasets:
            query = 'dataset dataset=/{}/{}*/NANOAODSIM'.format(dataset,year_tags[year])
            samples = get_das(query,verbose=verbose)
            if not samples: continue
            thesedatasets = set(s.split('/')[1] for s in samples)
            for thisdataset in thesedatasets:
                # NOTE: manually remove QCD_Pt-
                if 'QCD_Pt-' in thisdataset: continue
                if thisdataset not in result: result[thisdataset] = {}
                sampleMap = result[thisdataset].get('files',{})
                goodsamples = []
                for sample in samples:
                    if not update and sample in sampleMap: continue
                    if 'Validation error' in sample: continue
                    if sample.split('/')[1]!=thisdataset: continue
                    query = 'file dataset={}'.format(sample)
                    sampleMap[sample] = get_das(query,verbose=verbose)
                    goodsamples += [sample]
    
                result[thisdataset] = {'datasets': goodsamples, 'files': sampleMap}
        dump(fname,result)

get_mc(update,verbose)
