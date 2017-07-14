# Coda: a convolutional denoising algorithm for genome-wide ChIP-seq data

Coda uses convolutional neural networks to learn a mapping from noisy to high-quality ChIP-seq data.
These trained networks can then be used to remove noise and improve the quality of new ChIP-seq data.
For more details, please refer to our paper

Koh PW, Pierson E, Kundaje A, Denoising genome-wide histone ChIP-seq with convolutional neural networks. Bioinformatics (2017) 33 (14): i225-i233 URL:https://doi.org/10.1093/bioinformatics/btx243 (ISMB 2017 Proceedings)

bioRxiv doi: https://doi.org/10.1101/052118


## Dependencies
The code is written in Python 2.7 and requires the following Python packages to run:
- Numpy (1.11.1)
- Scipy (0.18.0)
- Scikit-learn (0.17.1)
- Pandas (0.18.1)
- h5py (2.6.0)
- rpy2 (2.8.1)
- Keras (1.0.7)

In addition, if you want to process your own data, you will need:
- AQUAS ChIP-seq pipeline
- SAMtools (1.2)
- BEDtools (2.23)
- ucsc_tools (3.0.9)

## Training and testing a model with pre-processed data
The fastest way to get started is to download data that has already been pre-processed. 
We have uploaded processed ChIP-seq data from lymphoblastoid cell lines GM12878 and GM18526, 
taken from [1]. Each cell line has two sets of ChIP-seq data, one derived from 1M reads per mark and 
the other from 100M+ reads per mark. The instructions below will train a model to recover high-depth
data from low-depth data on GM12878, and then apply it to low-depth data on GM18526, evaluating the 
model output against high-depth data on GM18526:

1) Clone the repo and install the dependencies above.

2) Edit `diConstants.py` to reflect the paths where you want to store the data, code, results, etc.

3) Run `setup.py`. This runs a few test imports to make sure you have the required libraries, and sets
up the directory structure as specified in `diConstants.py`.

4) Run `copyData.py`. This copies the required data (including hg19 blacklist and chromosome sizes) to 
the appropriate folders. Note that the data is 6GB in size, so please run this script in a location
where there's enough space!

5) Finally, run `python runGMExperiments.py` to get the experiments going. Numerical results will be 
written to `RESULTS_ROOT`. Output tracks (reconstructed signal and peak calls) will be written to `RESULTS_BIGWIG_ROOT`.
We make use of the R 'PRROC' package, written by Jan Grau and Jens Keilwagen, to evaluate peak calls.

## Processing your own data
We use the AQUAS ChIP-seq pipeline (https://github.com/kundajelab/TF_chipseq_pipeline)
to process raw ChIP-seq data. The script `prepData.py` (and the contents of the `scripts` folder)
contains wrapper functions that call the AQUAS pipeline for you. 

Please install the AQUAS pipeline before proceeding. Note that this pipeline is still under
some development and might be changing in non-backwards-compatible ways. Our code has been tested with
commit 7b7dd27d42d46ac52f5687f80904c576d1b6595d of the AQUAS pipeline. 

To create the processed data that we provided above, you may run the following steps:

1) Follow steps 1-3 of the above section.

2) Download the files corresponding to GM12878 and GM18526:
http://gbsc-share.stanford.edu/chromovar/rawdata/mapped/bam/personal/reconcile/dedup/

3) Run `python prepData.py make_intervals hg19`. You only need to do this once.

4) Run `python prepData.py run_GM_pipeline`. 

This code assumes that you've downloaded the files to a shared location 
(`REMOTE_ROOT`, specified in diConstants.py). It makes copies of the files in a 
local directory, `RAW_ROOT`, before proceeding. This setup is useful if `REMOTE_ROOT`
is shared across multiple machines and `RAW_ROOT` is local to the machine that you're
running the code on, because there will be a lot of IO operations that will be faster
if done locally. If you do not need this, modify `merge_BAMs()` in `prepData.py`
to remove the copying.

To process your own data, simply modify the paths in `diConstants.py` or copy your 
data to the right directories. While we start from BAM files in this example, the AQUAS 
pipeline can start from a variety of input files (e.g., FASTQ, tagAligns). Edit 
`scripts/getSignalTrack.sh` and `scripts/findPeaks.sh` if you want to change the parameters that 
are passed into AQUAS.

## Contact
If you have any questions, please contact:
- Pang Wei Koh <pangwei@cs.stanford.edu>
- Emma Pierson <emmap1@stanford.edu>
- Anshul Kundaje <akundaje@stanford.edu>

## References
[1] Kasowski M, Kyriazopoulou-Panagiotopoulou S, Grubert F, Zaugg JB, Kundaje A, Liu Y, et al. Extensive variation in chromatin states across humans. Science (New York, NY). 2013 11;342(6159):750–2
