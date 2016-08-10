import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.metrics import precision_recall_curve
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP
import datetime
from random import sample

def get_MSE(pred_Y, test_Y):
    """
    Returns mean squared error calculated across all dimensions.
    """
    assert pred_Y.shape == test_Y.shape
    return np.mean((pred_Y - test_Y) ** 2)

def get_pearsonR(pred_Y, test_Y):
    """
    Returns Pearson correlation for a single mark.
    
    Only takes in vectors.
    """

    assert pred_Y.shape == test_Y.shape
    assert len(pred_Y.shape) == 1
    
    return pearsonr(pred_Y, test_Y)[0]

def is_binary(M):
    unique_elements = list(set(M.flatten()))
    return all([elem in [0, 1] for elem in unique_elements])

def downsample_curve(vals):
    """
    Downsamples vals by a factor of 10 if len(vals) > 1000 (used to keep precision / recall curves from getting too long)
    """
    n = len(vals)
    
    if n > 1000:
        new_vals = []
        for i in range(int(n / 10)):
            new_vals.append(vals[i * 10])
        return new_vals
    else:
        return list(vals)

def compute_recalls_at_precision(precisions, recalls):
    """
    Computes recalls at 10%, 20%, ... 90% precision. 
    Does not interpolate.
    """
    precision_increment = .1
    desired_precision = precision_increment
    desired_precisions = []
    recalls_at_precision = []
    for i in range(len(precisions)):
        while precisions[i] > desired_precision:
            desired_precisions.append(desired_precision)
            recalls_at_precision.append(recalls[i])
            desired_precision += precision_increment

    return desired_precisions, recalls_at_precision


def compare(pred_Y, test_Y, predict_binary_output, peaks=None,
            save_curves=True, save_data=False):
    """
    Evaluates performance for predictions pred_Y relative to true labels test_Y. 
    If predict_binary_output, pred_Y should be a set of scores and test_Y should be 0, 1 labels. 
    Otherwise, both pred_Y and test_Y should be continuous values. 
    Returns squared error and Pearson correlation between the predicted output and the actual output.
    
    Both pred_Y and test_Y must be matrices of shape num_examples x num_histone_marks, 
    or they must both be matrices of shape num_examples x seq_length x num_histone_marks.
    If the latter, examples are concatenated together before correlations are computed.

    peaks is a list. Each element of this list corresponds to one mark and is a N x 2 matrix 
    where each row contains the (start, end) coordinates of a peak in that mark.
    If passing in peaks, make sure the coordinate system matches that of pred_Y and test_Y!
    For example, if your peaks start at the start of the chromosome, then pred_Y and test_Y have
    to start at the start of the chromosome as well.

    If save_curves is True, it saves the full precision-recall curve. save_curves cannot be True if 
    predict_binary_output is False. Right now it saves recalls @10, 20...90% precision. 

    If save_data is True, it saves the first mark of pred_Y and test_Y.
        
    Returns results, a dictionary containing:
        'AUC' (if predict_binary_output)
        'AUPRC' (if predict_binary_output)
        'precision_curves' (if save_curves)
        'recall_curves' (if save_curves)
        'threshold_curves' (if save_curves)
        'MSE' (if not predict_binary_output)
        'true_var' (if not predict_binary_output)
        'pearsonR' (if not predict_binary_output)
        'pred_Y' (if save_data)
        'test_Y' (if save_data)

    AUC, AUPRC, MSE, true_var, pearsonR, and spearmanR are each vectors of length num_histone_marks.  
    true_var is the variance of the true data; it is useful for interpreting whether a given
    MSE is good or bad.
    """
    
    # save_curves has to be False if predict_binary_output is also False
    if not predict_binary_output: save_curves = False
    
    pred_Y_is_binary = is_binary(pred_Y)
    test_Y_is_binary = is_binary(test_Y)   
    assert pred_Y.shape == test_Y.shape, \
        "pred_Y.shape = %s doesn't match test_Y.shape = %s" % (str(pred_Y.shape), str(test_Y.shape))
    assert test_Y_is_binary == predict_binary_output 

    #test_Y (the true labels) ought to be binary IFF we're predicting binary output. 
    #pred_Y should be a set of continuous scores, regardless of whether we're predicting binary output. 
    assert len(pred_Y.shape) == 2 or len(pred_Y.shape) == 3

    # If peaks is not None, then there should be one element in peaks for each mark in pred_Y.
    if peaks:
        assert len(peaks) == pred_Y.shape[-1]
    
    # If the input matrices are 3D, then squash the first two dimensions together
    if len(pred_Y.shape) == 3:
        pred_Y = np.reshape(pred_Y, [pred_Y.shape[0] * pred_Y.shape[1], pred_Y.shape[2]])
        test_Y = np.reshape(test_Y, [test_Y.shape[0] * test_Y.shape[1], test_Y.shape[2]])
    
    num_histone_marks = pred_Y.shape[len(pred_Y.shape) - 1]

    true_var = []
    MSE = []
    pearsonR = []

    precision_curves = []
    recall_curves = [] 
    threshold_curves = []
    auc = []
    auprc = []
    Y_pos_frac = []

    with open('PRROC.R', 'r') as f:#load in the R code. 
        r_fxn_string = f.read()
    r_auc_func = STAP(r_fxn_string, "auc_func")
    
    for mark_idx in range(num_histone_marks):
        ### Sub-select only peak regions
        if peaks:
            # If peaks exists but peaks[mark_idx] is set to None, we should skip this mark. 
            # This mark should correspond to INPUT, which has no peaks of its own.
            if peaks[mark_idx] is None:
                if predict_binary_output:
                    precision_curves.append(None)
                    recall_curves.append(None)
                    threshold_curves.append(None)
                    auprc.append(None)
                    auc.append(None)
                else:
                    true_var.append(None)
                    MSE.append(None)
                    pearsonR.append(None)
                continue

            # Initialize peak_idxs to all False
            num_bins = pred_Y.shape[0]
            peak_idxs = np.zeros(
                num_bins,
                dtype=bool)
            
            # Set peak_idx such that it is True in each peak
            # Simultaneously get the average signal density in each peak
            for peak_counter, peak in enumerate(peaks[mark_idx]):            
                # We have to check for this, because pred_Y and test_Y might only represent
                # a fraction of any given chromosome
                if peak[1] > num_bins: 
                    continue
                
                peak_idxs[peak[0]:peak[1]] = True                                

            pred_Y_mark = pred_Y[peak_idxs, mark_idx]
            test_Y_mark = test_Y[peak_idxs, mark_idx]
        else:
            pred_Y_mark = pred_Y[:, mark_idx]
            test_Y_mark = test_Y[:, mark_idx]

        ### Run evaluations on (selected) regions
        if predict_binary_output:
            precisions, recalls, thresholds = precision_recall_curve(test_Y_mark, pred_Y_mark)
            precisions, recalls = compute_recalls_at_precision(precisions, recalls)

            precision_curves.append(list(precisions))
            recall_curves.append(list(recalls))

            if len(test_Y_mark) < 100000:
                downsample_idxs = range(len(test_Y_mark))
            else:
                downsample_idxs = sample(range(len(test_Y_mark)), 100000)
    
            r_auprc_results = r_auc_func.pr_curve(scores_class0 = robjects.vectors.FloatVector(pred_Y_mark[downsample_idxs]), weights_class0 = robjects.vectors.FloatVector(test_Y_mark[downsample_idxs]))

            auprc.append(float(r_auprc_results.rx('auc.davis.goadrich')[0][0]))
            r_auc_results = r_auc_func.roc_curve(scores_class0 = robjects.vectors.FloatVector(pred_Y_mark[downsample_idxs]), weights_class0 = robjects.vectors.FloatVector(test_Y_mark[downsample_idxs]))
            auc.append(float(r_auc_results.rx('auc')[0][0]))
            Y_pos_frac.append(test_Y_mark.mean())        
            print("AUC %2.3f; AUPRC %2.3f" % (auc[mark_idx], auprc[mark_idx]))
        else:
            true_var.append(np.var(test_Y_mark))
            MSE.append(get_MSE(pred_Y_mark, test_Y_mark))
            pearsonR.append(get_pearsonR(pred_Y_mark, test_Y_mark))

            print("MSE %2.3f (true var %2.3f), pearsonR %2.3f" % 
                (MSE[mark_idx], true_var[mark_idx], pearsonR[mark_idx]))      

    if predict_binary_output:
        assert((len(precisions) > 0) and (len(recalls) > 0))
        results = {
            'AUC':auc,
            'AUPRC':auprc,
            'Y_pos_frac':Y_pos_frac
        }
        results['precision_curves'] = precision_curves
        results['recall_curves'] = recall_curves

    else:
        results = {
            'MSE': MSE,
            'true_var': true_var,
            'pearsonR': pearsonR
        }

    if save_data: 
        results['pred_Y'] = list(pred_Y[..., 0])
        results['test_Y'] = list(test_Y[..., 0])

    return results




