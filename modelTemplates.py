import copy
from dataset import Dataset, get_species_from_dataset_name
from diConstants import (HG19_ALL_CHROMS, MM9_ALL_CHROMS,
    HG19_TRAIN_CHROMS, MM9_TRAIN_CHROMS,
    VALID_CHROMS, TEST_CHROMS) 

def make_dataset_params(num_train_examples,
                        seq_length,
                        train_dataset_name='GM12878_5+1marks-K4me3_all', 
                        test_dataset_name='GM19239_5+1marks-K4me3_all',
                        train_X_subsample_target_string='5e6',
                        train_Y_subsample_target_string=None,
                        test_X_subsample_target_string=None,
                        test_Y_subsample_target_string=None,                        
                        random_seed=0,                        
                        num_test_examples=None,                        
                        normalization='arcsinh',
                        peak_fraction=0.5,
                        only_chr1=True,
                        num_bins_to_test=1000000,                        
                        train_chroms=None,
                        test_chroms=None):
    """
    only_chr1 controls whether genome-wide prediction is done on the whole genome, or just
    on chr1 for speed.

    num_bins_to_test controls how many bins of each chromosome should be tested. If num_bins_to_test
    == 1000000, for example, then only the first 1M bins of each chromosome (or of chr1, if only_chr1 is 
    True) will be tested. Set num_bins_to_test to None to test the whole chromosome.
    """

    if num_test_examples is None:
        num_test_examples = num_train_examples 

    if test_X_subsample_target_string is None:
        test_X_subsample_target_string = train_X_subsample_target_string
    
    if test_Y_subsample_target_string is None:
        test_Y_subsample_target_string = train_Y_subsample_target_string        

    if train_chroms is None:
        if get_species_from_dataset_name(train_dataset_name) == 'mm9':
            train_chroms = MM9_ALL_CHROMS
        else:
            train_chroms = HG19_ALL_CHROMS

    if test_chroms is None:
        if get_species_from_dataset_name(test_dataset_name) == 'mm9':
            test_chroms = MM9_ALL_CHROMS
        else:
            test_chroms = HG19_ALL_CHROMS

    return {
        'train_dataset': Dataset(
            dataset_name=train_dataset_name,
            num_examples=num_train_examples, 
            X_subsample_target_string=train_X_subsample_target_string, 
            Y_subsample_target_string=train_Y_subsample_target_string,
            random_seed=random_seed, 
            normalization=normalization,
            peak_fraction=peak_fraction,
            chroms=train_chroms),
        'test_datasets': [Dataset(
            dataset_name=test_dataset_name,
            num_examples=num_test_examples, 
            X_subsample_target_string=test_X_subsample_target_string, 
            Y_subsample_target_string=test_Y_subsample_target_string,
            random_seed=random_seed, 
            normalization=normalization,
            peak_fraction=peak_fraction,
            chroms=test_chroms)],
        'seq_length': seq_length,        
        'num_bins_to_test': num_bins_to_test,
        'only_chr1': only_chr1,
    }


def make_model_params(model_library,
                      model_class,
                      model_type,
                      dataset_params,       
                      scale_input='01',                                            
                      model_specific_params=None,
                      compile_params=None,
                      train_params=None,               
                      input_marks=None,
                      output_marks=None,
                      random_seed=0,
                      generate_bigWig=False,
                      predict_binary_output=False, 
                      zero_out_non_bins=False):    
    """
    input_marks is a list of histone marks that the model will take in as input.
    
    output_marks is a list of all the marks that we want the model to learn to output. 
    If we're training a single multi-task model, this is either a list of length 5 or 6, 
    depending on whether we're doing classification or regression (if we're doing classification, 
    we don't predict INPUT).
    If we're training a separate model for each mark, then output_marks is just a list of length 1.
    
    scale_input is one of 'ZCA', 'Z', '01', or 'identity'.  

    zero_out_non_bins is only used when predict_binary_output is True. It specifies whether 
    we should zero out the -log10 p values of bins that are not in the corresponding gappedPeak file.
    This is used for baseline evaluations.
    """
    
    params = {
        'model_library': model_library,
        'model_class': model_class,
        'model_type': model_type,
        'scale_input': scale_input,
        'random_seed': random_seed,
        'generate_bigWig': generate_bigWig,
        'predict_binary_output': predict_binary_output,
        'zero_out_non_bins': zero_out_non_bins
    }

    params['dataset_params'] = make_dataset_params(**dataset_params)

    # Defaults for compile_params
    if compile_params is None:
        compile_params = {}
    if model_library == 'keras':
        if predict_binary_output:
            compile_params_defaults = {
                'loss': 'binary_crossentropy',
                'optimizer': 'adagrad'
            }
        else:
            compile_params_defaults = {
                'loss': 'MSE',
                'optimizer': 'adagrad'
            }
        for key in compile_params_defaults:
            if key not in compile_params:
                compile_params[key] = compile_params_defaults[key]
    params['compile_params'] = compile_params

    # Defaults for train_params
    if train_params is None:
        train_params = {}
    if model_library == 'keras':
        train_params_defaults = {
            'nb_epoch': 50,
            'batch_size': 2000,
            'validation_split': 0.2
        }
        for key in train_params_defaults:
            if key not in train_params:
                train_params[key] = train_params_defaults[key]
    params['train_params'] = train_params

    # If input_marks is not set, then set it to all the marks in the training dataset
    if input_marks is None:
        input_marks = params['dataset_params']['train_dataset'].marks_in_dataset

    # Default for output_marks is to output all of the input_marks
    # Unless we're doing classification, in which case we don't output INPUT
    if output_marks is None:
        output_marks = copy.copy(input_marks)
        if predict_binary_output and 'INPUT' in output_marks:
            output_marks.remove('INPUT')

    # Make sure that input_marks and output_marks are both contained within
    # marks_in_train_dataset and marks_in_test_dataset
    for mark in input_marks + output_marks:
        assert mark in params['dataset_params']['train_dataset'].marks_in_dataset
        for test_dataset in params['dataset_params']['test_datasets']:
            assert mark in test_dataset.marks_in_dataset

    params['input_marks'] = input_marks
    params['output_marks'] = output_marks

    if model_specific_params is None:
        model_specific_params = {}
    for key in model_specific_params:
        if key in params:
            raise ValueError, 'model_specific_params cannot overwrite existing model params'
        params[key] = model_specific_params[key]

    return params

