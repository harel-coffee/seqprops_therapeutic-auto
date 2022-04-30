################################################################################
##  This script plots how F1 changes with respect to the decision threshold   ##
##  for all four models on a specified dataset.                               ##
################################################################################

import argparse
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import get_crossvalidation_f1
from matplotlib.pyplot import figure


# Parse arguments
parser = argparse.ArgumentParser(description='Plot how F1 changes with respect to the decision threshold for all four models on a specified dataset.')
parser.add_argument('dataset_csv', type=str, help='Filename of CSV file from datasets directory.')
args = parser.parse_args()
dataset_name = splitext(args.dataset_csv)[0]

# We performed 2 times 10-fold cross-validation which resulted in 100 runs
NB_RUNS = 20

# Set font size for plot
plt.rcParams.update({'font.size': 14})
figure(figsize=(8, 6), dpi=80)

# Define thresholds we will try
thresholds = np.linspace(0, 1, 100)

# Compute average F1 score for each threshold
scores_peptprop = [np.mean(get_crossvalidation_f1(dataset_name, "peptide_properties", nb_runs = NB_RUNS, threshold = threshold)) for threshold in thresholds]
scores_hot = [np.mean(get_crossvalidation_f1(dataset_name, "one_hot", nb_runs = NB_RUNS, threshold = threshold)) for threshold in thresholds]
scores_embedding = [np.mean(get_crossvalidation_f1(dataset_name, "embedding", nb_runs = NB_RUNS, threshold=threshold)) for threshold in thresholds]
scores_seqprop = [np.mean(get_crossvalidation_f1(dataset_name, "sequential_properties", nb_runs = NB_RUNS, threshold=threshold)) for threshold in thresholds]

# Plot dependency of F1 score on the chosen threshold
plt.plot(thresholds, scores_peptprop, label="Peptide properties")
plt.plot(thresholds, scores_hot, label="One-hot encoding")
plt.plot(thresholds, scores_embedding, label="Embedding")
plt.plot(thresholds, scores_seqprop, label="Sequential properties")
plt.xlabel("Decision threshold")
plt.ylabel("F1 score")
plt.legend(loc=0)
plt.tight_layout()
plt.show()