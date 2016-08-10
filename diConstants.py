import os

### Variables to set

# Where the remote directory that the GM12878 and GM18526 data have been downloaded to
REMOTE_ROOT = "/mnt/data/chromatinVariation1/rawdata/mapped/bam/personal/reconcile/dedup"    

# Where the AQUAS pipeline is installed
PIPELINE_ROOT = "/users/pangwei/TF_chipseq_pipeline/"

# Where the code is
CODE_ROOT = "/users/pangwei/deepimpute_pub"

# Where the bulk of the storage will be (intermediate/processed files, etc.)
DISK_ROOT = "/srv/scratch/pangwei/deepimpute_pub/"

# Where output bigwigs will be written to
RESULTS_BIGWIG_ROOT = "/srv/www/kundaje/deepimpute/model-bw"

HG19_BLACKLIST_FILE = '/srv/www/kundaje/pangwei/coda_denoising/hg19_blacklist.bed'
MM9_BLACKLIST_FILE = '/srv/www/kundaje/pangwei/coda_denoising/mm9_blacklist.bed'
HG19_CHROM_SIZES_PATH = '/srv/www/kundaje/pangwei/coda_denoising/hg19.chrom.sizes'
MM9_CHROM_SIZES_PATH = '/srv/www/kundaje/pangwei/coda_denoising/mm9.male.chrom.sizes'

MAPQ_THRESHOLD = 30

######


DATA_ROOT = os.path.join(DISK_ROOT, 'data')
MODELS_ROOT = os.path.join(DISK_ROOT, 'models')
RESULTS_ROOT = os.path.join(DISK_ROOT, 'results')

RAW_ROOT = os.path.join(DATA_ROOT, 'raw')
MERGED_ROOT = os.path.join(DATA_ROOT, 'merged')
SUBSAMPLED_ROOT = os.path.join(DATA_ROOT, 'subsampled')
BIGWIGS_ROOT = os.path.join(DATA_ROOT, 'bigWigs')
INTERVALS_ROOT = os.path.join(DATA_ROOT, 'intervals')
NUMPY_ROOT = os.path.join(DATA_ROOT, 'numpy')
PEAK_BASE_DIR = os.path.join(DATA_ROOT, 'peaks')
PEAK_GAPPED_DIR = os.path.join(PEAK_BASE_DIR, 'peak', 'macs2', 'rep1')
DATASETS_ROOT = os.path.join(DATA_ROOT, 'datasets')
BASE_ROOT = os.path.join(DATASETS_ROOT, 'base')
BASE_BIGWIG_ROOT = os.path.join(BASE_ROOT, 'bigWigs')
SEQ_ROOT = os.path.join(DATASETS_ROOT, 'processed-seq')

WEIGHTS_ROOT = os.path.join(MODELS_ROOT, 'weights')

LOSS_ROOT = os.path.join(RESULTS_ROOT, 'loss')
HIST_ROOT = os.path.join(RESULTS_ROOT, 'hist')
EVAL_ROOT = os.path.join(RESULTS_ROOT, 'eval')


HG19_CHROM_SIZES = {
    'chr1':  249250621,
    'chr2':  243199373,
    'chr3':  198022430,
    'chr4':  191154276,
    'chr5':  180915260,
    'chr6':  171115067,
    'chr7':  159138663,
    'chr8':  146364022,
    'chr9':  141213431,
    'chr10': 135534747,
    'chr11': 135006516,
    'chr12': 133851895,
    'chr13': 115169878,
    'chr14': 107349540,
    'chr15': 102531392,
    'chr16': 90354753,
    'chr17': 81195210,
    'chr18': 78077248,
    'chr19': 59128983,
    'chr20': 63025520,
    'chr21': 48129895,
    'chr22': 51304566,
}

MM9_CHROM_SIZES = {
    'chr1':  197195432,
    'chr2':  181748087,
    'chr3':  159599783,
    'chr4':  155630120,
    'chr5':  152537259,
    'chr6':  149517037,
    'chr7':  152524553,
    'chr8':  131738871,
    'chr9':  124076172,
    'chr10': 129993255,
    'chr11': 121843856,
    'chr12': 121257530,
    'chr13': 120284312,
    'chr14': 125194864,
    'chr15': 103494974,
    'chr16': 98319150,
    'chr17': 95272651,
    'chr18': 90772031,
    'chr19': 61342430
}
BIN_SIZE = 25
GENOME_BATCH_SIZE = 50000
NUM_BASES = 4

GM_CELL_LINES = ['GM12878', 'GM19239', 'GM10847', 'GM18505', 'GM18526', 'GM18951', 'GM2610']
GM_FACTORS = ['H3K27AC','H3K27ME3', 'H3K36ME3','H3K4ME1', 'H3K4ME3', 'INPUT']
SUBSAMPLE_TARGETS = ['0.1e6','0.25e6', '0.5e6','1e6', '2.5e6', '5e6','7.5e6', '10e6','30e6','20e6', None]

GM_DATASET_NAME_TEMPLATE = '%s_5+1marks-K4me3_all'
ROADMAP_DATASET_NAME_TEMPLATE = '%s_6+1marks_all'
ULI_DATASET_NAME_TEMPLATE = '%s_3marks_all'
MOW_DATASET_NAME_TEMPLATE = '%s_2marks_all'


HG19_ALL_CHROMS = [
    'chr1',
    'chr2',
    'chr3',
    'chr4',
    'chr5', 
    'chr6', 
    'chr7',
    'chr8', 
    'chr9', 
    'chr10', 
    'chr11', 
    'chr12', 
    'chr13', 
    'chr14', 
    'chr15', 
    'chr16', 
    'chr17', 
    'chr18', 
    'chr19', 
    'chr20', 
    'chr21', 
    'chr22', 
    ]

MM9_ALL_CHROMS = [
    'chr1',
    'chr2',
    'chr3',
    'chr4',
    'chr5', 
    'chr6', 
    'chr7',
    'chr8', 
    'chr9', 
    'chr10', 
    'chr11', 
    'chr12', 
    'chr13', 
    'chr14', 
    'chr15', 
    'chr16', 
    'chr17', 
    'chr18', 
    'chr19'
    ]


TEST_CHROMS = [
    'chr1',
    'chr2',
    ]

VALID_CHROMS = [
    'chr3',
    'chr4'
    ]

HG19_TRAIN_CHROMS = [
    'chr5', 
    'chr6', 
    'chr7',
    'chr8', 
    'chr9', 
    'chr10', 
    'chr11', 
    'chr12', 
    'chr13', 
    'chr14', 
    'chr15', 
    'chr16', 
    'chr17', 
    'chr18', 
    'chr19', 
    'chr20', 
    'chr21', 
    'chr22', 
    ]

MM9_TRAIN_CHROMS = [
    'chr5', 
    'chr6', 
    'chr7',
    'chr8', 
    'chr9', 
    'chr10', 
    'chr11', 
    'chr12', 
    'chr13', 
    'chr14', 
    'chr15', 
    'chr16', 
    'chr17', 
    'chr18', 
    'chr19'
    ]