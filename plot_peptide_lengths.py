#####################################################################################
##  This script creates a histogram of peptide lengths within a specified dataset  ##
#####################################################################################

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Creates a histogram of peptide lengths within a specified dataset')
parser.add_argument('dataset_csv', type=str)
args = parser.parse_args()
data = pd.read_csv("datasets/{}".format(args.dataset_csv))

positive_lengths = []
negative_lengths = []
for row_idx in range(len(data)):
    seq = data["sequence"][row_idx]
    seq_length = len(seq)
    label = data["label"][row_idx]

    if label == 0:
        negative_lengths.append(seq_length)
    else:
        positive_lengths.append(seq_length)

max_seq_len = max(max(negative_lengths), max(positive_lengths)) + 1
positive_histogram = np.zeros((max_seq_len))
negative_histogram = np.zeros((max_seq_len))

for seq_length in negative_lengths:
    negative_histogram[seq_length] += 1

for seq_length in positive_lengths:
    positive_histogram[seq_length] += 1


# Plot
ind = np.array(list(range(max_seq_len)))
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_subplot(111)
ax.bar(ind, negative_histogram, 0.27, color='r')
ax.bar(ind + 0.27, positive_histogram, 0.27, color='b')
ax.set_xlabel("Peptide length")
ax.set_ylabel("Count")

plt.show()