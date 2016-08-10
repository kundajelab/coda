import copy
import json
import os
import pandas as pd
import numpy as np
from IPython import embed
from collections import Counter

from diConstants import (
    SEQ_ROOT, BIN_SIZE, NUM_BASES,
    HG19_ALL_CHROMS, MM9_ALL_CHROMS, HG19_TRAIN_CHROMS, MM9_TRAIN_CHROMS, VALID_CHROMS, TEST_CHROMS,
    HG19_CHROM_SIZES, MM9_CHROM_SIZES)
from prepData import get_metadata_path, input_not_before_end, get_base_path, get_peaks, get_blacklisted_locs


def load_chrom(data_path, chrom):
    """
    Loads the .npz file in data_path and returns data for a given chrom.
    chrom is like "chr1"
    """
    m = np.load(data_path)
    return m[chrom]


def get_species_from_dataset_name(dataset_name):
    if 'ULI' in dataset_name or 'MOUSE' in dataset_name:
        return 'mm9'
    else:
        return 'hg19'  


class DatasetEncoder(json.JSONEncoder):
    """
    Encodes Dataset objects in JSON.
    """
    def default(self, obj):        
        if isinstance(obj, Dataset):
            return obj.__dict__
        else:
            return super(DatasetEncoder, self).default(obj)


class Dataset(object):
    """
    Dataset objects have the following fields:
        dataset_name
        num_examples
        X_subsample_target_string (string like "5e6" or None)
        Y_subsample_target_string (string like "5e6" or None)
        random_seed
        normalization
        peak_fraction
        chroms
        chroms_string
    
    They support the following methods:
        get_subsample_target_string(self, X_or_Y)
        get_seq_dataset_path(self, seq_length, factor_for_peaks)
        load_seq_dataset(self, seq_length, input_marks, output_marks)
        load_binary_genome(self, X_or_Y, marks, only_chr1=False)
        load_genome(self, X_or_Y, marks, only_chr1=False, peaks=False)
    
    And the static method process_subsample_target_string.
    """

    @staticmethod
    def process_subsample_target_string(subsample_target_string):
        if subsample_target_string is None:
            return subsample_target_string
        elif subsample_target_string == 'None':
            return None
        else:
            return str(subsample_target_string)


    def __init__(self, dataset_name, num_examples,
                 X_subsample_target_string, Y_subsample_target_string,
                 random_seed, normalization, peak_fraction, chroms):
        self.dataset_name = dataset_name
        self.num_examples = num_examples
        self.X_subsample_target_string = Dataset.process_subsample_target_string(X_subsample_target_string)
        self.Y_subsample_target_string = Dataset.process_subsample_target_string(Y_subsample_target_string)
        self.random_seed = random_seed
        self.normalization = normalization
        self.peak_fraction = peak_fraction
        self.chroms = chroms

        self.species = get_species_from_dataset_name(self.dataset_name)
        if self.species == 'hg19':
            all_chroms = HG19_ALL_CHROMS
            train_chroms = HG19_TRAIN_CHROMS
        else:
            all_chroms = MM9_ALL_CHROMS
            train_chroms = MM9_TRAIN_CHROMS

        if self.chroms == all_chroms:
            self.chroms_string = ""
        elif self.chroms == TEST_CHROMS:
            self.chroms_string = "_chroms-test"
        elif self.chroms == train_chroms:
            self.chroms_string = "_chroms-train"
        elif self.chroms == VALID_CHROMS:
            self.chroms_string = "_chroms-valid"
        else:
            raise ValueError, "chroms must be ALL_CHROMS, TEST_CHROMS, TRAIN_CHROMS, or VALID_CHROMS"
    
        if (self.normalization not in ['arcsinh', 'log', None]):
            raise ValueError, "normalization must be 'arcsinh', 'log', or None"

        peak_fraction = float(peak_fraction)
        if peak_fraction < 0.0 or peak_fraction > 1.0:
            raise ValueError, "peak_fraction must be in [0, 1]"

        try:
            metadata_path = get_metadata_path(self.dataset_name, self.X_subsample_target_string, self.normalization)
            with open(metadata_path, 'r') as f:        
                metadata = json.loads(f.read())
                self.marks_in_dataset = metadata['factors_to_include']
                self.cell_line = metadata['cell_line']
        except IOError:
            raise IOError, "Dataset %s doesn't exist." % metadata_path

        try:
            # Sanity check to make sure that metadata is consistent with different subsample target string            
            metadata_path = get_metadata_path(self.dataset_name, self.Y_subsample_target_string, self.normalization)
            with open(metadata_path, 'r') as f:        
                metadata = json.loads(f.read())
                assert self.marks_in_dataset == metadata['factors_to_include']
                assert self.cell_line == metadata['cell_line']
        except IOError:
            raise IOError, "Dataset %s doesn't exist." % metadata_path


    def get_subsample_target_string(self, X_or_Y):
        assert X_or_Y in ["X", "Y"]
        if X_or_Y == "X":
            return self.X_subsample_target_string
        else:
            return self.Y_subsample_target_string


    def get_seq_dataset_path(self, seq_length, factor_for_peaks):
        """
        If factor_for_peaks is INPUT,
        that means that all marks in the dataset are used for peak enrichment,
        but that the Y matrices in the dataset only contain the INPUT mark.
        This is used for training a separate model that only outputs INPUT.

        In contrast, if factor_for_peaks is None, all marks in dataset are similarly
        used for peak enrichment, but the Y matrices in the dataset contain all marks.
        This is used for training a single, multi-task model that outputs all marks.

        Y_subsample_target_string is normally set to None, unless we are intentionally
        trying to use a certain subsampling depth as the "full" data. 
        """

        if factor_for_peaks is None:
            dataset_path = os.path.join(
                SEQ_ROOT, "%s_subsample-%s-%s_rS-%s_numEx-%s_seqLen-%s_peakFrac-%s_norm-%s%s.npz" % \
                (self.dataset_name, self.X_subsample_target_string, self.Y_subsample_target_string, 
                 self.random_seed, self.num_examples, 
                 seq_length, self.peak_fraction, self.normalization, self.chroms_string))
        else:
            dataset_path = os.path.join(
                SEQ_ROOT, "%s_subsample-%s-%s_rS-%s_numEx-%s_seqLen-%s_peakFrac-%s_peaksFac-%s_norm-%s%s.npz" % \
                (self.dataset_name, self.X_subsample_target_string, self.Y_subsample_target_string, 
                 self.random_seed, self.num_examples, 
                 seq_length, self.peak_fraction, factor_for_peaks, self.normalization, self.chroms_string))

        return dataset_path


    def load_seq_dataset(self, seq_length, input_marks, output_marks):
        """
        Reads in (X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY) from a previously created .npz file, 
        where X is the input (subsampled) and Y is the output (full) and 
        peakPValueX, peakPValueY contain the -log10 pvalues for the called peaks (bin by bin)
        peakBinaryX, peakBinaryY contain the binarized peak signal for the called peaks (bin by bin)
        X is of shape num_examples x seq_length x len(input_marks).
        Y is of shape num_examples x seq_length x len(output_marks).

        peakPValueX is of similar shape to X, except that it does not contain an INPUT track, so it is of shape    
        num_examples x seq_length x (len(input_marks) - ('INPUT' in input_marks)).
        peakPValueY is of similar shape to Y, except that it does not contain an INPUT track, so it is of shape    
        num_examples x seq_length x (len(output_marks) - ('INPUT' in output_marks)).

        input_marks is a list of marks that will be used as input to the model. 
        output_marks is a list of marks that will be used as output from the model. It can be of length 1-6, depending
        on whether we're training separate models or one single model, and on whether we're doing classification
        or regression.

        If the .npz file doesn't exist, it will create it by calling extract_seq_dataset.
        """
        assert(input_not_before_end(output_marks))
        assert(input_not_before_end(input_marks))

        for input_mark in input_marks:
            if input_mark not in self.marks_in_dataset:
                raise ValueError, "input_marks must be in marks_in_dataset"

        for output_mark in output_marks:
            if output_mark not in self.marks_in_dataset:
                raise ValueError, "output_marks must be in marks_in_dataset"

        # Construct an identifying string for this dataset based on what the output marks are.
        # If all marks in marks_in_dataset are present, then for brevity we omit output_marks_string. 
        if len(output_marks) == len(self.marks_in_dataset):
            output_marks_string = None
        else:
            output_marks_string = '-'.join(output_marks)

        dataset_path = self.get_seq_dataset_path(seq_length, output_marks_string)
        
        try:      
            with np.load(dataset_path) as data:
                X = data['X'].astype('float32')
                Y = data['Y'].astype('float32')
                peakPValueX = data['peakPValueX'].astype('float32')
                peakPValueY = data['peakPValueY'].astype('float32')
                peakBinaryX = data['peakBinaryX'].astype('int8')
                peakBinaryY = data['peakBinaryY'].astype('int8')

        except:
            print("Dataset %s doesn't exist or is missing a required matrix. Creating..." % dataset_path)
            
            X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY = self.extract_seq_dataset(
                seq_length,                
                output_marks,
                dataset_path)

        # Only select the input marks that we want
        marks_idx = []
        peak_marks_idx = []
        for mark in input_marks:
            marks_idx.append(self.marks_in_dataset.index(mark))

        # We don't want to have INPUT inside peakPValueX
        factors_without_input = copy.copy(self.marks_in_dataset)
        if 'INPUT' in factors_without_input:
            factors_without_input.remove('INPUT')

        for mark in input_marks:
            if mark == 'INPUT':
                continue
            peak_marks_idx.append(factors_without_input.index(mark))    

        X = X[..., marks_idx]
        peakPValueX = peakPValueX[..., peak_marks_idx]
        peakBinaryX = peakBinaryX[..., peak_marks_idx]

        assert(np.all(peakPValueX >= 0) & np.all(peakPValueY >= 0))

        if (X.shape[0], X.shape[1]) != (Y.shape[0], Y.shape[1]):
            raise Exception, "First two dimensions of X and Y shapes (num_examples, seq_length) \
                              need to agree."
        if (peakPValueX.shape[0], peakPValueX.shape[1]) != (peakPValueY.shape[0], peakPValueY.shape[1]):
            raise Exception, "First two dimensions of peakPValueX and peakPValueY shapes \
                              (num_examples, seq_length) need to agree."
        if len(peakPValueX) != len(X):
            raise Exception, "peakPValueX and X must have same length."
        
        if ((seq_length != X.shape[1]) or (seq_length != peakPValueX.shape[1])):
            raise Exception, "seq_length between model and data needs to agree"

        return X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY


    def extract_seq_dataset(self, seq_length, output_marks, dataset_path):
        """
        Returns (X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY), where X is the input (subsampled) and Y is the output (full)
        and peakPValueX and peakPValueY are the -log10 pvalue scores for peaks called using MACS on X and Y respectively. 
        Both X and Y are each of shape num_examples x seq_length x num_factors.
        peakPValueX and peakPValueY are each of shape num_examples x seq_length x num_factors - 1. (no peaks for input)

        Also writes all matrices to a compressed .npz file.

        peak_fraction is the fraction of examples that should be centered on a peak that exists in the full data.
        For example, if peak_fraction = 0.5, then half of the examples will have a peak at the 
        center of the sequence, and the other half will not.

        factor_for_peaks determines which factor is used for determining whether a given location is
        counted as a 'peak' or not, since it could be a peak in one factor but not another.
        It should be a string, like 'H3K27AC'. 
        If it is None (the singleton, not a string), then a location is counted as having a peak 
        so long as there's a peak in any factor.

        This function sets the numpy random seed.
        """

        def sanity_check():
            """
            Sanity checks on the full and subsampled data. 

            Uses full_path and sub_path as defined in the main function.
            """
            assert os.path.isfile(full_path), "%s does not exist" % full_path
            assert os.path.isfile(sub_path), "%s does not exist" % sub_path
            assert os.path.isfile(full_peak_path), "%s does not exist" % full_peak_path
            assert os.path.isfile(sub_peak_path), "%s does not exist" % sub_peak_path

            with np.load(full_path) as full_data:
                with np.load(sub_path) as sub_data:

                    full_chroms = full_data.keys()
                    sub_chroms = sub_data.keys()

                    assert set(full_chroms) == set(sub_chroms), \
                      "Full and subsampled data must have exactly the same chromosomes."

                    assert full_chroms == sub_chroms, \
                      "Technically this is ok, but it's weird that the chromosomes in the full and subsampled \
                      data are not in the same order."

                    for chrom in full_chroms:
                        assert full_data[chrom].shape == sub_data[chrom].shape, \
                          "Each chromosome should have exactly the same number of bins and factors in both \
                          datasets."

                    assert len(set([full_data[chrom].shape[1] for chrom in full_chroms])) == 1, \
                      "Number of factors should be constant across all chromosomes."

        def get_start_positions(data_path, cell_line, chroms):
            """
            Returns a dictionary where each chromosome is a key, and each value is a set
            of start positions in that chromosome from which we can extract an example.
            Chromosomes are chosen uniformly at random, so longer chromosomes are not sampled 
            more than shorter chromosomes.

            Start positions are chosen to be enriched in peaks in the full data, as specified
            by the peak_fraction parameter to the main extract_seq_dataset function.

            Uses seq_length and num_examples from the main function parameters.
            """
            assert(seq_length % 2 == 1)
            with np.load(data_path) as data:
                # Make sure all chroms are present in the data
                assert all([chrom in data.keys() for chrom in chroms])

                # How long is each chromosome?
                num_bins = {chrom: data[chrom].shape[0] for chrom in chroms}

            # Filter out blacklisted bins. Add a bit of buffer to be safe. 
            blacklist_buffer = 5
            blacklisted_locs = get_blacklisted_locs(cell_line)
            non_blacklisted_bins = {}
            for chrom in chroms:
                good_locs = np.ones(num_bins[chrom] - seq_length + 1, dtype=bool)
                print('Prior to filtering out bad locations for chromosome %s, %i bins available' % (chrom, len(good_locs)))
                for bad_range in blacklisted_locs[chrom]:
                    left = max(bad_range[0] - seq_length - blacklist_buffer, 0)
                    right = max(bad_range[1] + blacklist_buffer, 0)
                    good_locs[left:right] = 0        
                print('After filtering out bad locations for chromosome %s, %i bins available' % (chrom, len(good_locs)))
                non_blacklisted_bins[chrom] = np.flatnonzero(good_locs).tolist()

            # Which chromosome? Sample uniformly at random
            # without caring about chromosome length.
            # Then count how many samples we are getting from each chromosome.
            chrom_samples = list(
                np.random.choice(
                    chroms,
                    num_examples,
                    replace=True))
            num_samples = {chrom: chrom_samples.count(chrom) for chrom in chroms}

            # Load all peaks into memory
            print("Preparing peaks...")
            peaks = {}

            # We want to enrich the data with parts of the genome that have peaks in the output marks
            # We don't have INPUT peaks, so we remove it
            if output_marks == ['INPUT']:
                factors_for_peaks = copy.copy(marks_in_dataset)
            else:
                factors_for_peaks = copy.copy(output_marks)
            if 'INPUT' in factors_for_peaks:
                factors_for_peaks.remove('INPUT')
        
            # Get peaks that correspond to the "full" data as specified by Y_subsample_target_string
            for factor in factors_for_peaks:
                peaks[factor], _ = get_peaks(cell_line, factor, Y_subsample_target_string)
        
            # Get start positions from each chromosome
            start_positions = {}
            for chrom in chroms:

                print("Calculating start positions for %s" % chrom)
                # We shift each peak such that the peak will be in the middle of the example,
                # i.e., the start position is (seq_length - 1)/2 bins before the actual peak
                # Unless doing so would move the starting position off the actual chromosome
                # For example, say seq_length = 101 (so shift = 50) and there is a peak at position 1050.
                # We would include position 1000 = 1050 - shift as a start position. 
                # A sequence that starts at position 1000 would go to position 1100 (inclusive)
                # and the midpoint of that sequence, position 1050, would have a peak.
                shift = int((seq_length - 1) / 2)
                peak_bins = np.zeros(max(non_blacklisted_bins[chrom]), dtype=bool)

                for factor in factors_for_peaks:
                    num_peaks = 0
                    for peak in peaks[factor][chrom]:
                        
                        left = max(int(peak[0] - shift), 0)
                        right = max(int(peak[1] - shift), 0)
                        num_peaks += right - left

                        # A "1" in peak_bins means that starting at that location will result 
                        # in a sequence whose center is a peak.
                        peak_bins[left:right] = 1
                    print("    %s peaks for %s" % (num_peaks, factor))
                peak_bins_binarized = np.copy(peak_bins)
                peak_bins = set(np.flatnonzero(peak_bins).tolist())
                    
                # Remove blacklisted bins, and create nonpeak_bins
                all_bins = set(non_blacklisted_bins[chrom])
                peak_bins = peak_bins.intersection(all_bins)
                nonpeak_bins = all_bins.difference(peak_bins)
                peak_bins = list(peak_bins)
                nonpeak_bins = list(nonpeak_bins)
                print("    Total after blacklisting: %s peaks and %s non-peaks" % (len(peak_bins), len(nonpeak_bins)))

                # Get samples of peak and non-peak locations
                peak_samples = np.round(num_samples[chrom] * peak_fraction).astype(int)
                nonpeak_samples = num_samples[chrom] - peak_samples

                start_positions[chrom] = np.random.choice(
                    nonpeak_bins,
                    nonpeak_samples,
                    replace=False)
                

                # There is a potential problem here if we are trying to draw more peak_samples
                # than there are peak locations on the chromosome.
                # If so, np.random.choice will error out.
                start_positions[chrom] = np.concatenate([
                    start_positions[chrom],
                    np.random.choice(
                        peak_bins,
                        peak_samples,
                        replace=False)])
                
                # Sort in the hopes that it makes memory access in extract_single_dataset faster
                start_positions[chrom].sort()
                
                
            return start_positions


        def extract_single_dataset(data_path, start_positions, marks):
            """
            From the data in data_path, extracts num_examples subsequences of length seq_length 
            from the start positions in start_positions.

            Returns a matrix of size num_examples x seq_length x num_marks.

            Uses seq_length and num_examples from the main function parameters.
            
            Used to load both the continuous signal and the peak p-values.
            """
            print('Extracting samples from %s...' % data_path)
            
            num_marks = len(marks)
            marks_idx = []
            for mark in marks:
                marks_idx.append(
                    marks_in_dataset.index(mark))
            
            return_dataset = np.empty([num_examples, seq_length, num_marks])
            first_empty_row = 0

            with np.load(data_path) as data:
                # Get required samples from each chromosome
                for chrom in start_positions.keys():
                                
                    data_chrom = data[chrom] 

                    for start_pos in start_positions[chrom]:                
                        return_dataset[first_empty_row, :, :] = data_chrom[
                            start_pos : start_pos+seq_length, 
                            marks_idx]
                        first_empty_row += 1

                    print("At sample number %s..." % first_empty_row)

            assert first_empty_row == num_examples

            # Note: this dataset has not been randomized yet. 
            # So it has consecutive elements from the same chromosome. 
            # We will randomize both X and Y datasets together later.
            return return_dataset


        def extract_binary_peak_dataset(full_path, subsample_target_string_to_extract, start_positions,
                                        cell_line, marks):
            """
            Method for returning Y with peak information. A 1 denotes a peak. 
            From the data in data_path, extracts num_examples subsequences of length seq_length 
            from the start positions in start_positions.

            Returns binary_peak_matrix, a matrix of size num_examples x seq_length x num_marks,
            where a 1 denotes a peak        
            """

            shift = int((seq_length - 1) / 2)
            peak_pval_matrix = np.empty([
                num_examples, 
                seq_length, 
                (len(marks) - ('INPUT' in marks))
            ])

            factor_idx = 0
            for factor in marks:
                if factor == 'INPUT':
                    continue
                first_empty_row = 0
                peak_dict, peak_log_pvalue_dict = get_peaks(
                    cell_line, 
                    factor, 
                    subsample_target_string=subsample_target_string_to_extract)
                for chrom in start_positions:
                    peak_vector_length = max(
                        np.max(peak_dict[chrom]), 
                        np.max(start_positions[chrom]) + seq_length) + 1
                    peak_pval_vector = np.zeros([peak_vector_length,])
                    for peak_idx, peak in enumerate(peak_dict[chrom]):
                        peak_pval_vector[peak[0]:peak[1]] = peak_log_pvalue_dict[chrom][peak_idx]
                    is_peak = (peak_pval_vector > 0)
                    print(factor, chrom, is_peak[start_positions[chrom] + shift].mean())
                    for start_pos in start_positions[chrom]:
                        peak_pval_matrix[first_empty_row, :, factor_idx] = peak_pval_vector[start_pos : (start_pos+seq_length)]
                        first_empty_row += 1            
                factor_idx += 1          
            binary_peak_matrix = (peak_pval_matrix > 0) * 1.
            assert np.all(peak_pval_matrix >= 0) 
            return binary_peak_matrix


        def extract_single_sequence_dataset(start_positions):
            """
            Extracts num_examples subsequences of length seq_length, at positions start_positions,
            from the hg19 sequence.

            Returns a matrix of size num_examples x (seq_length * BIN_SIZE) x NUM_BASES.

            Uses seq_length and num_examples from the main function parameters.
            """
            print('Extracting sequences...')
                    
            return_dataset = np.empty([num_examples, BIN_SIZE*seq_length, NUM_BASES])
            first_empty_row = 0
        
            # Get required samples from each chromosome
            for chrom in start_positions.keys():
                            
                data_chrom = load_seq_for_chrom(chrom)

                for start_pos in start_positions[chrom]:                
                    return_dataset[first_empty_row, :, :] = data_chrom[
                        start_pos*BIN_SIZE : (start_pos+seq_length)*BIN_SIZE, :]
                    first_empty_row += 1

                print("At sample number %s..." % first_empty_row)

            assert first_empty_row == num_examples

            # Make sure each base has at most one 1, and that at least one base is not N
            assert np.max(np.sum(return_dataset, axis=2)) == 1

            return return_dataset


        ### Main function code starts here

        # Read dataset metadata 
        dataset_name = self.dataset_name
        X_subsample_target_string = self.X_subsample_target_string
        Y_subsample_target_string = self.Y_subsample_target_string
        random_seed = self.random_seed
        num_examples = self.num_examples
        peak_fraction = self.peak_fraction
        normalization = self.normalization
        marks_in_dataset = self.marks_in_dataset
        cell_line = self.cell_line
        chroms = self.chroms

        # We always prepare dataset files with the full set of input_marks
        input_marks = copy.copy(marks_in_dataset)

        np.random.seed(random_seed)

        full_path = get_base_path(dataset_name, Y_subsample_target_string, normalization)
        sub_path = get_base_path(dataset_name, X_subsample_target_string, normalization)
        full_peak_path = get_base_path(dataset_name, Y_subsample_target_string, normalization=None, peaks=True)
        sub_peak_path = get_base_path(dataset_name, X_subsample_target_string, normalization=None, peaks=True)

        print('input', input_marks)
        print('output', output_marks)
        print('sub path', sub_path)
        print('full path', full_path)
        print('sub peak path', sub_peak_path)
        print('full peak path', full_peak_path)

        # Sanity check the input
        sanity_check()

        # Get a shared list of start positions for both X and Y
        # then extract the datasets
        start_positions = get_start_positions(full_path, cell_line, chroms)
        X = extract_single_dataset(sub_path, start_positions, input_marks)
        peakPValueX = extract_single_dataset(
            sub_peak_path, 
            start_positions, 
            [a for a in input_marks if a != 'INPUT'])
        peakBinaryX = extract_binary_peak_dataset(
            sub_path, 
            X_subsample_target_string, 
            start_positions, 
            cell_line, 
            [a for a in input_marks if a != 'INPUT'])
        
        Y = extract_single_dataset(full_path, start_positions, output_marks)
        peakPValueY = extract_single_dataset(
            full_peak_path, 
            start_positions, 
            [a for a in output_marks if a != 'INPUT'])
        peakBinaryY = extract_binary_peak_dataset(
            full_path, 
            Y_subsample_target_string, 
            start_positions, 
            cell_line, 
            [a for a in output_marks if a != 'INPUT'])

        
        # Sanity check the output
        assert (X.shape[0], X.shape[1]) == (Y.shape[0], Y.shape[1])
        assert (peakPValueX.shape[0], peakPValueX.shape[1]) == (peakPValueY.shape[0], peakPValueY.shape[1])
        assert X.shape[2] == len(input_marks)
        assert peakPValueX.shape[2] + ('INPUT' in input_marks) == len(input_marks)
        assert Y.shape[2] == len(output_marks)
        assert peakPValueY.shape[2] + ('INPUT' in output_marks) == len(output_marks)
        assert(peakPValueY.shape == peakBinaryY.shape)
        assert(peakPValueX.shape == peakBinaryX.shape)

        assert X.shape[0] == num_examples
        assert X.shape[1] == seq_length
        assert peakPValueX.shape[0] == num_examples
        assert peakPValueX.shape[1] == seq_length
        
        assert np.all(peakPValueX >= 0)
        assert np.all(peakPValueY >= 0)

        # If we only have one output mark, make sure the peak fraction is close. 
        if len(output_marks) == 1 and output_marks != ['INPUT']:
            midpoint = (seq_length - 1) / 2#
            true_peak_fraction = peakBinaryY[:, midpoint, 0].mean()

            assert np.abs(true_peak_fraction - peak_fraction) < 1e-2, 'Error: true peak fraction is %2.3f, desired fraction is %2.3f' % (true_peak_fraction, peak_fraction)


        # Randomize the ordering of return_dataset so we don't see consecutive elements 
        # from the same chromosome
        random_ordering = np.random.permutation(X.shape[0])
        X = X[random_ordering]
        Y = Y[random_ordering]
        peakPValueX = peakPValueX[random_ordering]
        peakPValueY = peakPValueY[random_ordering]
        peakBinaryX = peakBinaryX[random_ordering]
        peakBinaryY = peakBinaryY[random_ordering]

        # Downsize
        X = X.astype('float32')
        Y = Y.astype('float32')
        peakPValueX = peakPValueX.astype('float32')
        peakPValueY = peakPValueY.astype('float32')
        peakBinaryX = peakBinaryX.astype('int8')
        peakBinaryY = peakBinaryY.astype('int8')



        # Write output to disk
        print("Writing output to %s" % dataset_path)
        
        np.savez_compressed(
            dataset_path, 
            X=X, 
            Y=Y, 
            peakPValueX=peakPValueX, 
            peakPValueY=peakPValueY,
            peakBinaryX=peakBinaryX,
            peakBinaryY=peakBinaryY)

        return (X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY)


    def load_genome(self, X_or_Y, marks, only_chr1=False, peaks=False):
            """
            Loads a genome with the appropriate normalization, selecting only chroms present 
            in self.chroms.

            The only_chr1 flag is provided for convenience, so that code runs faster when we are only 
            looking at chr1.

            If peaks = True, loads the peak p-values instead. 
            """
            subsample_target_string = self.get_subsample_target_string(X_or_Y)
            
            data_path = get_base_path(self.dataset_name, subsample_target_string, self.normalization, peaks=peaks)

            # We only want to return the tracks corresponding to marks
            # The genome file has all factors in marks_in_dataset, so we iterate through marks
            # to pick out the correct indices
            if peaks and ('INPUT' in self.marks_in_dataset):
                marks_in_dataset = copy.copy(self.marks_in_dataset)
                marks_in_dataset.remove('INPUT')
            marks_idx = []

            for mark in marks:
                assert mark in self.marks_in_dataset
                marks_idx.append(self.marks_in_dataset.index(mark))

            with np.load(data_path) as data:
                # We have to create a new dictionary for the returned data
                # because data is a NpzFile object that does not support item assignment
                # We index with marks_idx so that only the correct tracks are returned.
                return_data = {} 
                if only_chr1 == False:
                    for key in self.chroms:
                        return_data[key] = data[key][..., marks_idx]
                else:
                    return_data['chr1'] = data['chr1'][..., marks_idx]

            for key in return_data:
                assert len(return_data[key].shape) == 2
                assert return_data[key].shape[1] == len(marks)

            return return_data


    def load_binary_genome(self, X_or_Y, marks, only_chr1=False):
        """
        Loads a binary genome, selecting only chroms present in self.chroms.
        Returns peak_matrices, peak_pval_matrices
        where peak_pval_matrices is a dictionary where each key is a chromosome, value is a matrix
        which is chrom_length x len(marks) with a zero if there's no peak and a -log10 pvalue otherwise.
        peak_matrices is the same but with a 1, not a p-value, for peaks; returned for convenience. 

        normalization is passed in only to get the correct metadata.
        """
        subsample_target_string = self.get_subsample_target_string(X_or_Y)

        peak_dict = {}
        peak_pval_dict = {}

        for mark in marks:
            peak_dict[mark], peak_pval_dict[mark] = get_peaks(self.cell_line, mark, subsample_target_string)
            
        peak_matrices = {}
        peak_pval_matrices = {}

        if self.species == 'hg19':
            chrom_sizes = HG19_CHROM_SIZES
        else:
            chrom_sizes = MM9_CHROM_SIZES
        chroms_to_use = self.chroms if not only_chr1 else ['chr1']

        for chromosome in chroms_to_use:
            n_bins_in_chrom = int(chrom_sizes[chromosome] / 25.)
            peak_matrices[chromosome] = np.zeros([n_bins_in_chrom, len(marks)])
            peak_pval_matrices[chromosome] = np.zeros([n_bins_in_chrom, len(marks)])
            for mark_idx, mark in enumerate(marks):
                for i, peak in enumerate(peak_dict[mark][chromosome]):
                    peak_matrices[chromosome][peak[0]:peak[1], mark_idx]  = 1
                    peak_pval_matrices[chromosome][peak[0]:peak[1], mark_idx]  = peak_pval_dict[mark][chromosome][i]
        
        return peak_matrices, peak_pval_matrices
