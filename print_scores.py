############################################################################################
##  This script computes different scores for all four models on a specified dataset.     ##
##  Note that ROC-AUC is computed by plot_roc.py script.                                  ##
############################################################################################

import argparse
from os.path import splitext
from helper_functions import load_predicted_probabilities, load_true_labels, geometric_mean_score
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Plot F1 scores as box plots for all four models on a specified dataset.')
parser.add_argument('dataset_csv', type=str, help='Filename of CSV file from datasets directory.')
args = parser.parse_args()
dataset_name = splitext(args.dataset_csv)[0]

# We performed 2 times 10-fold cross-validation which resulted in 20 runs
NB_RUNS = 20

def print_scores(y_trues, y_preds):
    # Apply threshold of 0.5 to y_preds
    for y_pred in y_preds:
        for idx, val in enumerate(y_pred):
            if val > 0.5:
                y_pred[idx] = 1
            else:
                y_pred[idx] = 0
    print("F1 score: {:.3f}".format(np.mean([f1_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])))
    print("MCC: {:.3f}".format(np.mean([matthews_corrcoef(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])))
    print("GM: {:.3f}".format(np.mean([geometric_mean_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])))
    print("Recall: {:.3f}".format(np.mean([recall_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])))
    print("Precision: {:.3f}".format(np.mean([precision_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)])))

#########################################################
## Print scores for each model and plot its ROC curve  ##
#########################################################
# MLP with peptide properties
y_trues = load_true_labels(dataset_name, "peptide_properties", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "peptide_properties", nb_runs = NB_RUNS)
print("Peptide properties")
print_scores(y_trues, y_preds)

# RNN with One-Hot Vector Encoding
y_trues = load_true_labels(dataset_name, "one_hot", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "one_hot", nb_runs = NB_RUNS)
print()
print("One-hot vector encoding")
print_scores(y_trues, y_preds)

# RNN with Embedding
y_trues = load_true_labels(dataset_name, "embedding", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "embedding", nb_runs = NB_RUNS)
print()
print("Embedding")
print_scores(y_trues, y_preds)

# RNN with Sequential properties
y_trues = load_true_labels(dataset_name, "sequential_properties", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "sequential_properties", nb_runs = NB_RUNS)
print()
print("Sequential properties")
print_scores(y_trues, y_preds)