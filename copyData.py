import os
from subprocess import call

import diConstants as di

call('wget http://mitra.stanford.edu/kundaje/pangwei/coda_denoising/hg19_blacklist.bed', shell=True)
call('wget http://mitra.stanford.edu/kundaje/pangwei/coda_denoising/hg19.chrom.sizes', shell=True)
call('mv hg19_blacklist.bed %s' % di.HG19_BLACKLIST_FILE, shell=True)
call('mv hg19.chrom.sizes %s' % di.HG19_CHROM_SIZES_PATH, shell=True)

call('wget http://mitra.stanford.edu/kundaje/pangwei/coda_denoising/low_seq_depth_processed_files.tar.gz', shell=True)
call('tar -xvf low_seq_depth_processed_files.tar.gz', shell=True)
call('mv *metadata *npz %s' % di.BASE_ROOT, shell=True)
call('mv *gappedPeaks* %s' % di.PEAK_GAPPED_DIR, shell=True)