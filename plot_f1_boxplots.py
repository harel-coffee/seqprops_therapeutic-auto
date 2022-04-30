############################################################################################
##  This script plots F1 scores as box plots for all four models on a specified dataset.  ##
############################################################################################

import argparse
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import get_crossvalidation_f1
from matplotlib.pyplot import figure


# Parse arguments
parser = argparse.ArgumentParser(description='Plot F1 scores as box plots for all four models on a specified dataset.')
parser.add_argument('dataset_csv', type=str, help='Filename of CSV file from datasets directory.')
args = parser.parse_args()
dataset_name = splitext(args.dataset_csv)[0]

# We performed 2 times 10-fold cross-validation which resulted in 100 runs
NB_RUNS = 20

# Set font size for plot
plt.rcParams.update({'font.size': 14})
figure(figsize=(8, 6), dpi=80)

data = [
    get_crossvalidation_f1(dataset_name, "peptide_properties", nb_runs = NB_RUNS, threshold = 0.5),
    get_crossvalidation_f1(dataset_name, "one_hot", nb_runs = NB_RUNS, threshold = 0.5),
    get_crossvalidation_f1(dataset_name, "embedding", nb_runs = NB_RUNS, threshold = 0.5),
    get_crossvalidation_f1(dataset_name, "sequential_properties", nb_runs = NB_RUNS, threshold = 0.5),
]

bp = plt.boxplot(data)
ax = plt.gca()
plt.xlim((0.7, 4.7))
plt.xticks([1, 2, 3, 4], ["Peptide\nproperties", "One-hot", "Embedding", "Sequential\nproperties"])
plt.ylabel("F1 score")

# Add mean and variance to the plot
for i, line in enumerate(bp['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ²={:.2e}'.format(np.mean(data[i]), np.var(data[i])).replace("e-0", "e-")
    ax.annotate(text, xy=(x + 0.035, y), rotation=90)

plt.show()