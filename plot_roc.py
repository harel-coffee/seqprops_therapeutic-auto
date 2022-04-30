################################################################################
##  This script plots ROC curves for all four models on a specified dataset.  ##
##  ROC-AUC value is also shown on the plot.                                  ##
################################################################################

import argparse
from os.path import splitext
from sklearn import metrics
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from helper_functions import load_true_labels, load_predicted_probabilities
from matplotlib.pyplot import figure

# Parse arguments
parser = argparse.ArgumentParser(description='Plot ROC curves for all four models on a specified dataset.')
parser.add_argument('dataset_csv', type=str, help='Filename of CSV file from datasets directory.')
args = parser.parse_args()
dataset_name = splitext(args.dataset_csv)[0]

# We performed 2 times 10-fold cross-validation which resulted in 20 runs
NB_RUNS = 20

# Set font size for plot
plt.rcParams.update({'font.size': 14})
figure(figsize=(8, 6), dpi=80)


# See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
def plot_avg_roc(label, y_trues, y_preds):
    x_axis = np.linspace(0, 1, 10000)
    aucs = []
    tprs = []

    for y_true, y_pred in zip(y_trues, y_preds):
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        interp_tpr = np.interp(x_axis, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(metrics.roc_auc_score(y_true, y_pred))
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(x_axis, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(
        x_axis,
        mean_tpr,
        label="{} (AUC={:.3f} $\pm$ {:.2f})".format(label, mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )




#########################################################
## Print scores for each model and plot its ROC curve  ##
#########################################################
# MLP with peptide properties
y_trues = load_true_labels(dataset_name, "peptide_properties", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "peptide_properties", nb_runs = NB_RUNS)
plot_avg_roc("Peptide properties", y_trues, y_preds)

# RNN with One-Hot Vector Encoding
y_trues = load_true_labels(dataset_name, "one_hot", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "one_hot", nb_runs = NB_RUNS)
plot_avg_roc("One-hot vector encoding", y_trues, y_preds)

# RNN with Embedding
y_trues = load_true_labels(dataset_name, "embedding", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "embedding", nb_runs = NB_RUNS)
plot_avg_roc("Embedding", y_trues, y_preds)

# RNN with Sequential properties
y_trues = load_true_labels(dataset_name, "sequential_properties", nb_runs = NB_RUNS)
y_preds = load_predicted_probabilities(dataset_name, "sequential_properties", nb_runs = NB_RUNS)
plot_avg_roc("Sequential properties", y_trues, y_preds)

# Show ROC curves and AUC values
plt.xlabel("False-positive rate")
plt.ylabel("True-negative rate")
plt.legend(loc=0)
plt.tight_layout()
plt.show()