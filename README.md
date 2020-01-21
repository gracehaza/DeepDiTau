# DeepDiTau
Training package for DeepDiTau tagging in CMSSW

```bash
source miniconda3/bin/activate
conda create -y -n ml_root_py36 python=3.6
source activate ml_root_py36
export CUDA_HOME=/cvmfs/cms-lpc.opensciencegrid.org/sl7/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
conda config --add channels conda-forge
conda install -y root
conda install -y cudatoolkit=9.2
conda install -y cudnn
conda install -y scikit-learn
conda install -y mkl
conda install -y uproot
conda install -y pandas
pip install --upgrade tensorflow-gpu==1.12
pip install --upgrade keras==2.2.4 # works with 1.12
pip install --upgrade matplotlib
pip install gpustat
pip install hyperopt
```
