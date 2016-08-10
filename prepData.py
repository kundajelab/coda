from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

from subprocess import call, check_output
import os
import json
from traceback import print_exc
import signal
import sys
import pandas as pd
from time import time, sleep
import numpy as np
import multiprocessing
import thread
import gzip

import IPython

from diConstants import (PIPELINE_ROOT, CODE_ROOT, DATA_ROOT, RAW_ROOT, MERGED_ROOT, REMOTE_ROOT,
    SUBSAMPLED_ROOT, BIGWIGS_ROOT, INTERVALS_ROOT, NUMPY_ROOT, BASE_ROOT, BASE_BIGWIG_ROOT,
    RESULTS_BIGWIG_ROOT, MODELS_ROOT,
    HG19_BLACKLIST_FILE, MM9_BLACKLIST_FILE,
    BIN_SIZE, HG19_CHROM_SIZES, HG19_CHROM_SIZES_PATH, MM9_CHROM_SIZES, MM9_CHROM_SIZES_PATH,
    PEAK_BASE_DIR, COMBINED_PEAK_DIR, SUBSAMPLE_TARGETS,
    GM_CELL_LINES, GM_FACTORS, GM_DATASET_NAME_TEMPLATE,        
    HG19_ALL_CHROMS, MM9_ALL_CHROMS,
    MAPQ_THRESHOLD)


def perform_normalization(X, normalization):
    """
    Normalizes a dataset using a method in ['log', 'arcsinh', None]. If none, just returns original dataset. 
    """
    assert(normalization in ['log', 'arcsinh', None])   
    if set(X.flatten()) == set([1.0, 0.0]):
        assert(normalization is None)

    if normalization in ['arcsinh', 'log']:
        if normalization == 'arcsinh':
            X = np.arcsinh(X)
        else:
            X = np.log(X + 1)
        print('Normalization: took %s of data. Mean is now %2.3f, max %2.3f' % (normalization, np.mean(X), np.max(X)))
    return X


def perform_denormalization(X, normalization):
    """
    Denormalizes a dataset using a method in ['log', 'arcsinh', None]. If none, just returns original dataset. 
    """

    assert(normalization in ['log', 'arcsinh', None])   
    if set(X.flatten()) == set([1.0, 0.0]):
        assert(normalization is None)
    if normalization in ['arcsinh', 'log']:
        if normalization == 'arcsinh':
            X = np.sinh(X)
        else:
            X = np.exp(X) - 1
        print('Denormalization: took inverse %s of data. Mean is now %2.3f, max %2.3f' % (normalization, np.mean(X), np.max(X)))
    return X


def check_npz_files():
    """
    This confirms that we can load all the .npz files in BASE_DIR (for some reason they were getting corrupted.)
    """
    desired_keys = ['chr' + str(i) for i in range(1, 23)]
    n_successes = n_errors = 0
    for f in os.listdir(BASE_ROOT):
        if '.npz' not in f:
            continue
        try:
            d = np.load(os.path.join(BASE_ROOT, f))
            n_successes += 1
            assert(sorted(d.keys()) == sorted(HG19_ALL_CHROMS)) # This check will fail on mouse
        except:
            n_errors += 1
            os.remove(os.path.join(BASE_ROOT, f))
            print('Error with ' + f)
            continue
    print('successes', n_successes, 'errors', n_errors)


def get_peaks(cell_line, factor, subsample_target_string):
    """
    chrs_to_peaks: a dictionary whose keys are chromosomes which map to an array of bin starts and ends 
    indices (not chromosome locations) which are peaks. 
    Eg, {'chr1':[[5, 10], [25, 50]]} means bins 5 - 9 and 25 - 49 on chromosome 1 are peaks. 
    When computing peak boundaries, rounds (ie, a peak beginning at bin .6 = a bin beginning at bin 1.)
    peak_log_pvalues: a dictionary whose keys are chromosomes which map to an array of peak log pvalues
    in the same order as the peaks in chrs_to_peaks. 
    Eg, {'chr1':[99, 104]}  means the peaks in chr1 have log10 pvalues 99 and 104, respectively. 
    """
    
    peak_path = get_peak_path(cell_line, factor, subsample_target_string)
    if not os.path.isfile(peak_path):
        raise ValueError, "%s does not exist." % peak_path

    d = pd.read_csv(peak_path, sep = '\t', header = None)
    d = d[[0, 1, 2, 13]]

    d.columns = ['chr', 'start', 'end', 'log10_pvalue']
    chrs = list(set(d['chr']))
    chrs_to_peaks = {}
    peak_log_pvalues = {}
    for chrom in chrs:
        idxs = d['chr'] == chrom
        chrs_to_peaks[chrom] = np.array(zip(list(d.loc[idxs]['start']), list(d.loc[idxs]['end'])))
        chrs_to_peaks[chrom] = np.around(chrs_to_peaks[chrom] / BIN_SIZE).astype(int)
        peak_log_pvalues[chrom] = np.array(d.loc[idxs]['log10_pvalue'])
        assert(len(peak_log_pvalues[chrom]) == len(chrs_to_peaks[chrom]))
    return chrs_to_peaks, peak_log_pvalues



def generate_bigWig(data, marks, bigWig_prefix, bigWig_folder):
    """
    Takes in data, a dictionary with keys corresponding to chromosomes
    and each chromosome being a matrix of shape num_bins x num_histone_marks
    and outputs bigWigs generated from that data in bigWig_folder,
    one for each factor in FACTORS_TO_INCLUDE
    """

    assert data[data.keys()[0]].shape[1] == len(marks)
    chrom_sizes_path = HG19_CHROM_SIZES_PATH

    for (factorIdx, factor) in enumerate(marks):
        
        wig_path = os.path.join(bigWig_folder, '%s_%s.wig' % (bigWig_prefix, factor))
        bigWig_path = os.path.join(bigWig_folder, '%s_%s.bw' % (bigWig_prefix, factor))

        with open(wig_path, 'w') as f:
            for chrom in data:

                f.write('fixedStep chrom=%s start=1 step=%s span=%d\n' % (chrom, BIN_SIZE, BIN_SIZE))

                for i in data[chrom][:, factorIdx]:                    
                    f.write('%s\n' % str(i))
                    
        call('bash scripts/convertWigToBigWig.sh %s %s %s' % (wig_path, bigWig_path, chrom_sizes_path), 
             shell=True)        

    return None
        

def get_blacklisted_locs(cell_line):
    """
    Returns a dictionary whose keys are chromosomes which map to an array of bin starts and ends 
    indices (not chromosome locations) to exclude: does not include upper end of range (in line with numpy indexing conventions).
    Eg, {'chr1':[[5, 10], [25, 50]]} means we should exclude bins 5 - 9 and 25 - 49 on chromosome 1.
    """
    if get_species(cell_line) == 'mm9':
        blacklist_file = MM9_BLACKLIST_FILE
    else:
        blacklist_file = HG19_BLACKLIST_FILE

    d = pd.read_csv(blacklist_file, sep = "\t")
    blacklist_dictionary = {}
    for i in range(len(d)):
        chrom = d.iloc[i]['chromosome']
        start = d.iloc[i]['start']
        end =  d.iloc[i]['end']
        if chrom not in blacklist_dictionary:
            blacklist_dictionary[chrom] = []
        blacklist_dictionary[chrom].append([int(1.*start / BIN_SIZE), int(1. * end / BIN_SIZE) + 1])

    return blacklist_dictionary


def get_merged_BAM_path(cell_line, factor):
    """
    Returns the path to the BAM file that contains all merged replicates 
    for a given cell_line and factor.
    """

    return os.path.join(MERGED_ROOT, '%s-%s_merged.bam' % (cell_line, factor))


def get_merged_BED_SE_path(cell_line, factor):
    """
    Returns the path to the BED file that contains all merged replicates 
    for a given cell_line and factor. This is for single-end reads.
    These BED files have already been filtered for MAPQ.
    """

    return os.path.join(MERGED_ROOT, '%s-%s_merged.bed' % (cell_line, factor))


def get_merged_BED_path(cell_line, factor):
    """
    Returns the path to the BEDPE file that contains all merged replicates 
    for a given cell_line and factor.
    These BEDPE files have already been filtered for MAPQ and properly paired reads.
    """

    return os.path.join(MERGED_ROOT, '%s-%s_merged.bedpe' % (cell_line, factor))


def get_tagAlign_path(cell_line, factor, subsample_target_string = None):
    """
    Returns the path to the tagAlign file that contains all merged replicates 
    for a given cell_line and factor.
    These tagAlign files have already been filtered for MAPQ and properly paired reads.

    If subsample_target_string is specified, return a subsampled tagAlign instead.
    """

    if subsample_target_string:
        return os.path.join(SUBSAMPLED_ROOT, '%s-%s_subsample-%s.tagAlign.gz' % (cell_line, factor, subsample_target_string))
    else:
        return os.path.join(MERGED_ROOT, '%s-%s_merged.tagAlign.gz' % (cell_line, factor))    


def get_bigWig_folder(cell_line, factor, subsample_target_string = None):
    """
    Returns the name of the output folder where bigWigs for a given cell_line, factor,
    and optionally subsample_target_string should be placed. 
    This output folder is passed to the ENCODE CHiP-seq pipeline.
    """

    if subsample_target_string:
        return os.path.join(BIGWIGS_ROOT, '%s-%s_subsample-%s' % (cell_line, factor, subsample_target_string))    
    else:
        return os.path.join(BIGWIGS_ROOT, '%s-%s_merged' % (cell_line, factor))    


def get_peak_path(cell_line, factor, subsample_target_string):
    assert(factor != 'INPUT')
    if subsample_target_string:
        subsample_output_string = "subsample-%s" % subsample_target_string
    else:
        subsample_output_string = "merged"

    return os.path.join(
        PEAK_BASE_DIR,
        'peak',
        'macs2',
        'rep1',
        '%s-%s_%s' % (cell_line, factor, subsample_output_string) + 
        '.tagAlign_x_%s-INPUT_%s.tagAlign.gappedPeak.gz' % (cell_line, subsample_output_string))


def get_peak_bigWig_path(cell_line, factor, subsample_target_string = None):
    """
    Returns the path to the bigWig file that contains the peak p-values
    for a given cell_line, factor, and optionally 
    subsample_target_string.
    """
    if subsample_target_string:
        subsample_output_string = "subsample-%s" % subsample_target_string
    else:
        subsample_output_string = "merged"

    return os.path.join(
        PEAK_BASE_DIR,
        'signal',
        'macs2',
        'rep1',
        '%s-%s_%s' % (cell_line, factor, subsample_output_string) + 
        '.tagAlign_x_%s-INPUT_%s.tagAlign.pval.signal.bw' % (cell_line, subsample_output_string))


def get_bigWig_path(cell_line, factor, subsample_target_string = None):
    """
    Returns the path to the bigWig file that contains the output of align2rawsignal 
    (from the ENCODE CHiP-seq pipeline) for a given cell_line, factor, and optionally 
    subsample_target_string.
    """
    
    if subsample_target_string:
        return os.path.join(
            BIGWIGS_ROOT, 
            '%s-%s_subsample-%s' % (cell_line, factor, subsample_target_string),
            'signal',
            'tag2bw',
            'rep1',
            '%s-%s_subsample-%s.bigwig' % (cell_line, factor, subsample_target_string))

    else:
        return os.path.join(
            BIGWIGS_ROOT, 
            '%s-%s_merged' % (cell_line, factor),
            'signal',
            'tag2bw',
            'rep1',
            '%s-%s_merged.bigwig' % (cell_line, factor))


def get_intervals_path(chrom, species):
    """
    Returns the path to the intervals BED file for a given chromosome.

    This BED file contains equally spaced intervals at BIN_SIZE."""

    assert species in ['hg19', 'mm9']
    return os.path.join(INTERVALS_ROOT, '%s_%s_%s.bed' % (species, chrom, BIN_SIZE))


def get_numpy_path(cell_line, factor, chrom, subsample_target_string=None):
    """
    Returns the path of the numpy array containing the binned signal for a given cell_line, factor,
    and optionally subsample_target_string.
    """
    
    if subsample_target_string:
        return os.path.join(NUMPY_ROOT, '%s-%s-%s_subsample-%s.npy' % (cell_line, factor, chrom, subsample_target_string))

    else:
        return os.path.join(NUMPY_ROOT, '%s-%s-%s_merged.npy' % (cell_line, factor, chrom))

def get_peak_numpy_path(cell_line, factor, chrom, subsample_target_string=None):
    """
    Returns the path of the numpy array containing the binned peak p-value signal for a given cell_line, factor,
    and optionally subsample_target_string.
    """
    assert(factor != 'INPUT')
    if subsample_target_string:
        return os.path.join(NUMPY_ROOT, 'peak_pvals_by_bin_%s-%s-%s_subsample-%s.npy' % (cell_line, factor, chrom, subsample_target_string))

    else:
        return os.path.join(NUMPY_ROOT, 'peak_pvals_by_bin_%s-%s-%s_merged.npy' % (cell_line, factor, chrom))

def get_base_path(dataset_name, subsample_target_string, normalization, peaks=False):
    """
    If peaks is True, returns the base path for the peak pvals; otherwise, returns base path for continuous signal.

    Normalization is always set to None if peaks is True. 
    """
    if peaks:
        return os.path.join(BASE_ROOT, 'peak_pvals_by_bin_%s_subsample-%s_norm-None.npz' % 
                        (dataset_name, subsample_target_string))
    else:
        return os.path.join(BASE_ROOT, '%s_subsample-%s_norm-%s.npz' % 
                        (dataset_name, subsample_target_string, normalization))


def get_metadata_path(dataset_name, subsample_target_string, normalization):
    return os.path.join(BASE_ROOT, '%s_subsample-%s_norm-%s.metadata' % 
                        (dataset_name, subsample_target_string, normalization))
    

def merge_BAMs(cell_lines_to_use, factors_to_use):
    """
    Takes a remote directory (REMOTE_ROOT) containing several different cell lines, marks, and 
    replicates, copies the data over to a local directory (RAW_ROOT), then combines all replicates 
    for each pair of cell lines and marks. Outputs to MERGED_ROOT.
    Only looks at cell lines that are in cell_lines_to_use and marks that are in factors_to_use.

    Operates on raw data available at http://gbsc-share.stanford.edu/chromovar/rawdata/
    """ 

    cell_mark_pairs = set()
    cell_mark_name_triples = [] 
    all_cmds = [[]]

    # First, copy files over from REMOTE_ROOT (/mnt/data...) to RAW_ROOT 
    for f in os.listdir(REMOTE_ROOT):
        if (os.path.isfile(os.path.join(REMOTE_ROOT, f)) and f.startswith('SNYDER_HG19_') 
            and f.endswith('.dedup.bam')):

            spl = f.split('_')
            cell_line = spl[2]
            if cell_line not in cell_lines_to_use:
                continue

            mark = spl[3]
            if mark not in factors_to_use:
                continue

            all_cmds[0].append('cp %s %s' % (os.path.join(REMOTE_ROOT, f), RAW_ROOT))

            cell_mark_pairs.add((cell_line, mark))
            cell_mark_name_triples.append((cell_line, mark, f))

    # Then process all files in RAW_ROOT
    for (cell, mark) in cell_mark_pairs:

        # How many replicates does this (cell, mark) pair have?
        count = 0
        filename = ''
        for (c, m, f) in cell_mark_name_triples:
            if cell == c and mark == m:
                count += 1
                filename = f
        assert count > 0
        
        if count == 1:
            print("%s-%s has no replicates. Copying straight..." % (cell, mark))
            all_cmds[-1].append("cp %s %s;" % (os.path.join(RAW_ROOT, filename), get_merged_BAM_path(cell, mark)))

        else:
            print("%s-%s has %s replicates. Merging..." % (cell, mark, count))
            all_cmds[-1].append("samtools merge %s %s/*%s_%s*.bam" % \
                (get_merged_BAM_path(cell, mark), RAW_ROOT, cell, mark))
    return all_cmds


def filter_and_convert_BAMs(cell_lines_to_use, factors_to_use):
    """
    Looks at all merged BAM files in MERGED_ROOT, and for each BAM file,
    filters out all reads below MAPQ 30 and all reads that aren't paired properly,
    and then outputs a tagAlign.gz file with only the filtered reads 
    in the same MERGED_ROOT folder.
    """
    all_cmds = [[], []]
    for cell_line in cell_lines_to_use:
        for factor in factors_to_use:

            BAM_path = get_merged_BAM_path(cell_line, factor)
            tagAlign_path = get_tagAlign_path(cell_line, factor)
            
            if os.path.isfile(BAM_path):        
                BED_path = get_merged_BED_path(cell_line, factor)
                all_cmds[0].append("bash scripts/filterAndConvertBAMs.sh %s %s %s" % (BAM_path, BED_path, MAPQ_THRESHOLD))
                all_cmds[1].append("bash scripts/convertBEDPEtoTagAlign.sh %s %s" % (BED_path, tagAlign_path))
            else:
                print("Warning: %s does not exist. Skipping..." % BAM_path)
    return all_cmds


def subsample_BAMs(cell_lines_to_use, factors_to_use, subsample_targets_to_use):
    """
    For each cell_line and factor, subsamples the corresponding BEDPE file to 
    the desired depths. Outputs in SUBSAMPLED_ROOT a tagAlign.gz file for each 
    (cell_line, factor, subsample_target) combination.
    """
    all_cmds = [[]]
    for cell_line in cell_lines_to_use:
        for factor in factors_to_use:
            
            subsample_input = get_merged_BED_path(cell_line, factor)
            full_reads = int(float(check_output('wc -l %s' % subsample_input, shell=True).split(' ')[0]))
            # subsample_command = ""

            for subsample_target_string in subsample_targets_to_use:

                if subsample_target_string == None:
                    continue

                subsample_target = int(float(subsample_target_string))
                
                if full_reads < subsample_target:
                    print("Warning: %s-%s only has %s read pairs, less than subsampling target of %s. Skipping..." %
                        (cell_line, factor, full_reads, subsample_target_string))
                    continue

                print("Subsampling %s-%s: %s read pairs from %s read pairs" % (cell_line, factor, subsample_target_string, full_reads))

                subsample_output = get_tagAlign_path(cell_line, factor, subsample_target_string)

                # if subsample_command != "":
                    # subsample_command += '; '

                cmd = "bash scripts/subsampleBEDPEs.sh %s %s %s" % (subsample_input, subsample_output, subsample_target)
                # subsample_command += cmd

                all_cmds[0].append(cmd)
            
            # subsample_command = "(" + subsample_command + ") &"
            #call(subsample_command, shell=True)
    return all_cmds  


def get_chrom_sizes(cell_line):
    if get_species(cell_line) == 'mm9':
        return MM9_CHROM_SIZES
    else:
        return HG19_CHROM_SIZES

def get_species(cell_line):    
    if 'MOUSE' in cell_line:
        return 'mm9'
    else:
        return 'hg19'

def get_signal_tracks(cell_lines_to_use, factors_to_use, subsample_targets_to_use):
    """
    Calls the ENCODE CHiP-seq pipeline on the tagAlign files for all 
    cell lines, factors, and subsample targets (including the full data).
    Outputs in BIGWIGS_ROOT a .bigWig file for each 
    (cell_line, factor, subsample_target) combination.
    """
    all_cmds = [[]]
    for cell_line in cell_lines_to_use:
        species = get_species(cell_line)
        for factor in factors_to_use:

            chrom_sizes = get_chrom_sizes(cell_line)

            # This gets signal tracks from both full and subsampled data
            # because None is an element of SUBSAMPLE_TARGETS
                        
            signal_command = ""

            for subsample_target_string in subsample_targets_to_use:
                tagAlign_path = get_tagAlign_path(cell_line, factor, subsample_target_string)
                bigWig_folder = get_bigWig_folder(cell_line, factor, subsample_target_string)

                if os.path.isfile(tagAlign_path):                    
                    files_already_exist = check_whether_BW_files_exist(
                        cell_line, 
                        factor, 
                        subsample_target_string, 
                        average_peaks=False)

                    if files_already_exist:
                        print('Bigwig files already exist for %s; skipping.' % bigWig_folder)
                    else:
                        print('Bigwig files DO NOT exist for %s; adding to tasks.' % bigWig_folder)
                        if signal_command != "":
                            signal_command += '; '
                        cmd = "bash scripts/getSignalTrack.sh %s %s %s %s" % (PIPELINE_ROOT, tagAlign_path, bigWig_folder, species)
                        signal_command += cmd
                        all_cmds[0].append(cmd)
                else:
                    print("Warning: %s does not exist. Skipping..." % tagAlign_path)
                
            signal_command = "(" + signal_command + ") &"
            
            #call(signal_command, shell=True)
    return all_cmds

def make_intervals(species):
    """
    Constructs BED files, one for each chromosome, each containing equally
    spaced intervals at BIN_SIZE.

    The third column of the BED file is exclusive, i.e., the interval is
    actually [start, end). So for a BIN_SIZE of size 25 the intervals will look like
        chr1 0 25
        chr2 25 50
        ...

    For convenience, here is the official documentation:
    
        chromEnd - The ending position of the feature in the chromosome or scaffold. 
        The chromEnd base is not included in the display of the feature. 
        For example, the first 100 bases of a chromosome are defined as 
        chromStart=0, chromEnd=100, and span the bases numbered 0-99.

    The fourth column (name) is added because bigWigAverageOverBed only accepts 
    BED files with 4 columns.

    We just truncate the end of the chromosome if it's not cleanly divisible
    by BIN_SIZE.
    """

    if species == 'hg19':
        chrom_sizes = HG19_CHROM_SIZES
    elif species == 'mm9':
        chrom_sizes = MM9_CHROM_SIZES
    else:
        raise ValueError, 'species must be hg19 or mm9'

    for chrom, chrom_size in chrom_sizes.items():
        print("Generating BED file for %s" % chrom)
        BED_path = get_intervals_path(chrom, species)

        with open(BED_path, 'w') as f:
            for start in range(0, chrom_size - BIN_SIZE + 1, BIN_SIZE):
                end = start + BIN_SIZE
                name = "%s-%s" % (chrom, start)
                f.write("%s\t%s\t%s\t%s\n" % (chrom, start, end, name))

def check_whether_BW_files_exist(cell_line, factor, subsample_target_string, average_peaks):
    """
    Checks whether bigwig files + the corresponding interval paths exist. 
    """
    
    allFilesExist = True

    if average_peaks:
        bigWig_path = get_peak_bigWig_path(cell_line, factor, subsample_target_string)
    else:
        bigWig_path = get_bigWig_path(cell_line, factor, subsample_target_string)
    if not (os.path.isfile(bigWig_path)):
        allFilesExist = False

    species = get_species(cell_line)
    chrom_sizes = get_chrom_sizes(cell_line)
    for chrom in chrom_sizes.keys():
        BED_path = get_intervals_path(chrom, species)        
        if not os.path.isfile(BED_path):  
            allFilesExist = False

    return allFilesExist

def get_average_signal_over_intervals(cell_lines_to_use, factors_to_use, subsample_targets_to_use, average_peaks = False):
    """    
    Averages the signal in the .bigWig files in BIGWIGS_ROOT into bins of BIN_SIZE.
    Outputs a .npy file in NUMPY_ROOT for each (cell_line, factor, subsample_target) 
    combination.

    This calls the bigWigAverageOverBed tool from UCSC tools and takes the mean0 column.

    This function does nothing if the .npy file in NUMPY_ROOT already exists.
    """
    all_cmds = [[], [], []]
    assert(input_not_before_end(factors_to_use))
    for cell_line in cell_lines_to_use:
        for factor in factors_to_use:
            if average_peaks and factor == 'INPUT':
                continue

            chrom_sizes = get_chrom_sizes(cell_line)
            species = get_species(cell_line)
            # This averages signal tracks from both full and subsampled data
            # because None is an element of subsample_targets_to_use
            for subsample_target_string in subsample_targets_to_use:
                allFilesExist = check_whether_BW_files_exist(cell_line, factor, subsample_target_string, average_peaks)
                if allFilesExist:
                    print('All files exist for %s, %s, %s, average_peaks = %s; averaging signal over intervals' % (cell_line, factor, subsample_target_string, average_peaks))
                    for chrom in chrom_sizes.keys():
                        BED_path = get_intervals_path(chrom, species)
                        if average_peaks:
                            bigWig_path = get_peak_bigWig_path(cell_line, factor, subsample_target_string)
                            numpy_path = get_peak_numpy_path(cell_line, factor, chrom, subsample_target_string)
                        else:
                            bigWig_path = get_bigWig_path(cell_line, factor, subsample_target_string)
                            numpy_path = get_numpy_path(cell_line, factor, chrom, subsample_target_string)
                        output_path = bigWig_path + '-%s_binned.out' % chrom
                        if os.path.isfile(numpy_path):#we've already done everything. 
                            print("Warning: %s already exists. Skipping..." % numpy_path)
                        else:
                            print("Numpy file does not exist; creating %s" % (numpy_path))
                            if os.path.isfile(output_path):
                                print("Warning: %s already exists. Skipping..." % output_path)
                            else:
                                cmd = "bash scripts/averageSignalTrack.sh %s %s %s" % (bigWig_path, BED_path, output_path)
                                all_cmds[0].append(cmd)
                            all_cmds[1].append('python prepData.py turn_into_numpy %s %s' % (output_path, numpy_path))
                            # Clean up intermediate output
                            all_cmds[2].append("rm -rf %s" % output_path)

                else:
                    print('Warning: not all files exist for %s, %s, %s, average_peaks = %s' % (cell_line, factor, subsample_target_string, average_peaks))

    return all_cmds

def turn_into_numpy(output_path, numpy_path):
    """
    Saves the output_path as a numpy_path. 
    """
    df = pd.read_csv(output_path, header = None)
    np.save(numpy_path, np.array(df))


def prep_dataset(dataset_name, cell_line, factors_to_include, chroms_to_include, 
                 subsample_targets, normalization, peak_dataset = False):
    """
    Cobbles together a single .npz file containing binned signals for a given cell_line, 
    list of factors, and list of chromosomes. There is one .npz file per 
    (cell_line, subsample_target, normalization) triplet.
    
    Output is a single .npz file in BASE_ROOT with name dataset_name.
    This .npz file contains one matrix for each chromosome. 
    Each matrix is of dimensions num_bins x num_factors,
    where num_bins is roughly floor(length of chromosome / BIN_SIZE),
    and num_factors is the length of factors_to_include.

    If peak_dataset = True, loads a peak dataset instead. 
    """

    if peak_dataset:
        assert(normalization is None)

    assert(input_not_before_end(factors_to_include))
    if peak_dataset:
        factors_to_include = np.copy(factors_to_include)
        if factors_to_include[-1] == 'INPUT':
            factors_to_include = factors_to_include[:-1]

    for subsample_target_string in subsample_targets:
        
        output_path = get_base_path(dataset_name, subsample_target_string, normalization)
        if os.path.isfile(output_path):
            print('Output file %s exists' % output_path)
            continue
        print("Preparing %s %s" % (dataset_name, subsample_target_string))
        # First make sure that all the numpy files we need exist
        do_files_exist = True
        for chrom in chroms_to_include:
            for factor in factors_to_include:
                if peak_dataset:
                    numpy_path = get_peak_numpy_path(cell_line, factor, chrom, subsample_target_string)
                else:
                    numpy_path = get_numpy_path(cell_line, factor, chrom, subsample_target_string)
                if not os.path.isfile(numpy_path):
                    print('Warning: %s does not exist' % numpy_path)
                    do_files_exist = False
                    break

        if not do_files_exist:
            print("Warning: not all .npy files are ready to make dataset %s for %s %s" % (dataset_name, cell_line, subsample_target_string))
            continue
 

        # Write dataset metadata to disk
        if not peak_dataset:
            metadata = {
                'dataset_name': dataset_name,
                'cell_line': cell_line,
                'factors_to_include': factors_to_include,
                'chroms_to_include': chroms_to_include,
                'subsample_targets': subsample_targets,
                'normalization': normalization
            }
            metadata_path = get_metadata_path(dataset_name, subsample_target_string, normalization)
            with open(metadata_path, 'w') as f:
                f.write(json.dumps(metadata))


        # Construct output matrix

        num_factors = len(factors_to_include)
        matrices = {}
        unnormalized_matrices = {}
        blacklist_buffer = 5
        blacklisted_locs = get_blacklisted_locs(cell_line)

        for chrom in chroms_to_include:

            print("... packing %s" % chrom)

            first_factor = True

            for (idx, factor) in enumerate(factors_to_include): 
                if peak_dataset:
                    numpy_path = get_peak_numpy_path(cell_line, factor, chrom, subsample_target_string)
                else:
                    numpy_path = get_numpy_path(cell_line, factor, chrom, subsample_target_string)

                assert os.path.isfile(numpy_path), "Error: %s is missing" % numpy_path

                # Each individual chrom-factor is a column vector
                arr = np.load(numpy_path)

                if first_factor:
                    first_factor = False
                    num_bins = len(arr)
                    chrom_matrix = np.empty([num_bins, num_factors])

                chrom_matrix[:, idx] = arr[:, 0]

            # Zero out blacklist regions. Add a bit of buffer to be safe.                         
            print('Before blacklisting %s, average signal is %s' % (chrom, np.mean(chrom_matrix)))            
            for bad_range in blacklisted_locs[chrom]:
                chrom_matrix[bad_range[0]-blacklist_buffer : bad_range[1]+blacklist_buffer, :] = 0                                    
            print('After blacklisting %s, average signal is %s' % (chrom, np.mean(chrom_matrix)))            
            
            # Save matrix for this chrom
            unnormalized_matrices[chrom] = chrom_matrix
            matrices[chrom] = perform_normalization(chrom_matrix, normalization)
        np.savez_compressed(output_path, **matrices)

        # Always save unnormalized bigWigs even if the actual data is normalized
        # because we don't want to view normalized bigWigs on the genome browser
        generate_bigWig(
            unnormalized_matrices, 
            factors_to_include,
            '%s_subsample-%s_norm-None' % (dataset_name, subsample_target_string), 
            BASE_BIGWIG_ROOT)


def prep_dataset_wrapper(dataset_name, cell_line, factors_string, subsample_target, normalization, peak_dataset):
    """
    This is just a wrapper to allow prep dataset to be called from the command line. 
    """

    if normalization == 'None':
        normalization = None
    if subsample_target == 'None':
        subsample_target = None
    assert(peak_dataset in ['True', 'False'])
    peak_dataset = peak_dataset == 'True'

    if get_species(cell_line) == 'mm9':
        all_chroms = MM9_ALL_CHROMS
    else:        
        all_chroms = HG19_ALL_CHROMS
    prep_dataset(dataset_name, cell_line, factors_string.split('-'), all_chroms, 
                 [subsample_target], normalization, peak_dataset)
    

def generate_datasets(cell_lines_to_use, dataset_name_template, factors_to_use, subsample_targets_to_use):
    """
    Calls prep_dataset on each cell_line, factor, and subsample_target; 
    Each dataset uses data from chr1-22 and all factors in factors_to_use.
    Also creates peak datasets.

    Output is in BASE_ROOT.
    """
    all_cmds = [[]]
    factors_string = '-'.join(factors_to_use)
    for cell_line in cell_lines_to_use:        
        for subsample_target in subsample_targets_to_use:
            all_cmds[0].append('python prepData.py ' \
                + ' prep_dataset_wrapper peak_pvals_by_bin_%s %s %s %s None True' % \
                (dataset_name_template % cell_line, cell_line, factors_string, subsample_target))
            all_cmds[0].append('python prepData.py ' \
                + ' prep_dataset_wrapper %s %s %s %s arcsinh False' % \
                (dataset_name_template % cell_line, cell_line, factors_string, subsample_target))
    return all_cmds


def call_all_peaks(cell_lines_to_use, factors_to_use, subsample_targets_to_use):
    """
    Calls the ENCODE CHiP-seq pipeline on the tagAlign files for all 
    cell lines, factors, and subsample targets (including the full data).
    Outputs in PEAK_BASE_DIR/peaks_macs2/true_replicates a gappedPeak.gz file for each 
    (cellLine, factor, subsampleTarget) combination. 
    """
    print('calling all peaks!!')
    all_cmds = [[]]
    for cell_line in cell_lines_to_use:
        species = get_species(cell_line)
        for factor in factors_to_use:
            if factor == 'INPUT':
                continue

            controls_and_inputs = []
            
            for subsample_target_string in subsample_targets_to_use:
                if check_whether_BW_files_exist(cell_line, factor, subsample_target_string, average_peaks = True):
                    print('%-8s %-8s %-8s peak files already exist, not regenerating' % (cell_line, factor, subsample_target_string))
                    continue

                else:
                    input_file = get_tagAlign_path(cell_line, factor, subsample_target_string = subsample_target_string)
                    control_input_file = get_tagAlign_path(cell_line, 'INPUT', subsample_target_string = subsample_target_string)
                    if os.path.exists(input_file) and os.path.exists(control_input_file):
                        print('%-8s %-8s %-8s peak files DO NOT exist, regenerating' % (cell_line, factor, subsample_target_string))
                        controls_and_inputs.append([input_file, control_input_file])
                    else:
                        print('%-8s %-8s %-8s input files DO NOT exist, cannot call peaks' % (cell_line, factor, subsample_target_string))
                        continue

            for input_file, control_input_file in controls_and_inputs:                
                
                if os.path.isfile(input_file) and os.path.isfile(control_input_file):

                    if os.path.isfile(control_input_file):                    
                        cmd = "bash scripts/findPeaks.sh %s %s %s %s %s" % (PIPELINE_ROOT, PEAK_BASE_DIR, input_file, control_input_file, species)                        

                    all_cmds[0].append(cmd)                        

                    print('Running command ', cmd)
                else:
                    print("Warning: input file %s or %s does not exist. Skipping..." % (input_file, control_input_file))
            
            
    return all_cmds


def input_not_before_end(list_of_marks):
    """
    Makes sure that INPUT does not occur before the last element of a list of marks. 
    """
    return ('INPUT' not in list_of_marks[:-1])
def callCommand(cmd):
    call(cmd, shell = True)
    sleep(3)

def fork_and_wait(n_proc, target, args=[]):
    """
    Fork n_proc processes, run target(*args) in each, and wait to finish.
    This is Nathan's method. 
    """
    if n_proc == 1:
        target(*args)
        return
    else:
        pids = []
        for i in xrange(n_proc):
            pid = os.fork()
            if pid == 0:
                try:
                    signal.signal(signal.SIGINT, handle_interrupt_signal)
                    target(*args)
                    os._exit(os.EX_OK)
                except Exception, inst:
                    print_exc()
                    config.log_statement( "Uncaught exception in subprocess\n" 
                                          + traceback.format_exc(), log=True)
                    os._exit(os.EX_SOFTWARE)
            else:
                pids.append(pid)
        try:
            while len(pids) > 0:
                ret_pid, error_code = os.wait()
                if ret_pid in pids:
                    pids.remove(ret_pid)
                if error_code != os.EX_OK: 
                    raise OSError, "Process '{}' returned error code '{}'".format(
                        ret_pid, error_code) 
        except KeyboardInterrupt:
            for pid in pids:
                try: os.kill(pid, signal.SIGHUP)
                except: pass
            raise
        except OSError:
            for pid in pids:
                try: os.kill(pid, signal.SIGHUP)
                except: pass
            raise
        return

class Counter(object):
    """
    Nathan's implementation of the Counter class; used for running multiple threads simultaneously.
    """
    def __init__(self, initval=0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()
    
    def return_and_increment(self):
        with self.lock:
            rv = self.val.value
            self.val.value += 1
        return rv
def handle_interrupt_signal(signum, frame):
    os._exit(os.EX_TEMPFAIL)


def run_in_parallel(task_name, n_proc, target, all_args):
    """
    Run target on each item in items.
    all_args should be a list of lists (where each element is one argument set).
    """
    if len(all_args) == 0:
        print("No tasks to run!")
        return
    curr_item = Counter()
    def worker():
        index = curr_item.return_and_increment()
        while index < len(all_args):
            args = all_args[index]
            sys.stdout.write('Now running %s, command %i / %i with %i processes; commands are %s\n' % (task_name, index + 1, len(all_args), n_proc, args))
            sleep(2)
            sys.stdout.flush()
            sys.stderr.flush()
            target(*args)
            index = curr_item.return_and_increment()
        return

    fork_and_wait(n_proc, worker)

def callCommand(cmd):
    call(cmd, shell = True)
    sleep(3)


def run_pipeline_commands(cell_lines_to_use, factors_to_use, subsample_targets_to_use, 
                          dataset_name_template, n_processes = 8, steps_to_skip = []):

    """
    Runs the full pipeline using n_processes. 
    Skips steps in steps_to_skip. 

    Each method returns a list of lists: each element in the outside list is a list of bash commands that can be run in parallel. 
    """
    
    # GM-specific processing    
    if cell_lines_to_use[0].startswith('GM'):
        if 'merge_bam' not in steps_to_skip:
            merge_bam_cmds = merge_BAMs(cell_lines_to_use, factors_to_use)
            for cmd_set in merge_bam_cmds:
                run_in_parallel('Merge BAM', n_processes, callCommand, [[cmd] for cmd in cmd_set])
        if 'filter_bam' not in steps_to_skip:
            filter_bam_cmds = filter_and_convert_BAMs(cell_lines_to_use, factors_to_use)
            for cmd_set in filter_bam_cmds:
                run_in_parallel('Filter BAM', n_processes, callCommand, [[cmd] for cmd in cmd_set])
        if 'subsample_bam' not in steps_to_skip:
            subsample_bam_cmds = subsample_BAMs(cell_lines_to_use, factors_to_use, subsample_targets_to_use)
            for cmd_set in subsample_bam_cmds:
                run_in_parallel('Subsample BAM', n_processes, callCommand, [[cmd] for cmd in cmd_set])

    # Common processing
    if 'get_signal_tracks' not in steps_to_skip:
        signal_track_cmds = get_signal_tracks(cell_lines_to_use, factors_to_use, subsample_targets_to_use)
        for cmd_set in signal_track_cmds:
            run_in_parallel('Get signal track', n_processes, callCommand, [[cmd] for cmd in cmd_set])
    if 'call_peaks' not in steps_to_skip:
        call_peak_cmds = call_all_peaks(cell_lines_to_use, factors_to_use, subsample_targets_to_use)
        for cmd_set in call_peak_cmds:
            run_in_parallel('Call peak', n_processes, callCommand, [[cmd] for cmd in cmd_set])

    if 'get_average_signal' not in steps_to_skip:
        
        get_average_signal_peaks_cmds = get_average_signal_over_intervals(cell_lines_to_use, factors_to_use, subsample_targets_to_use, average_peaks = True)
        get_average_signal_cmds = get_average_signal_over_intervals(cell_lines_to_use, factors_to_use, subsample_targets_to_use, average_peaks = False)

        for cmd_set in get_average_signal_cmds + get_average_signal_peaks_cmds:

            run_in_parallel('Average signal', n_processes, callCommand, [[cmd] for cmd in cmd_set])

    generate_all_dataset_cmds = generate_datasets(cell_lines_to_use, dataset_name_template, 
                                                  factors_to_use, subsample_targets_to_use)
    for cmd_set in generate_all_dataset_cmds:
            run_in_parallel('Generate dataset', n_processes, callCommand, [[cmd] for cmd in cmd_set])


def run_GM_pipeline():
    """
    Runs the full pipeline (starting from subsampling) to get many subsample targets for GM12878 
    and GM18526, and one subsample target for the other cell lines.
    """

    try:        
        run_pipeline_commands(
            ['GM12878', 'GM18526'],
            GM_FACTORS, 
            ['0.5e6', None], 
            GM_DATASET_NAME_TEMPLATE, 
            steps_to_skip=['merge_bam', 'filter_bam', 'subsample_bam', 'get_signal_tracks', 'call_peaks'],
            n_processes=12)

    except:
        print_exc()
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == '__main__':
    """
    Calls a method using arguments from command line. Eg, 

    python prepData.py run_in_parallel a b c 

    calls run_in_parallel(a, b, c)
    """

    args = sys.argv
    fxn_args = args[2:]   
    print('Calling %s with arguments' % args[1], args[2:])
    locals()[args[1]](*args[2:])
   
    
