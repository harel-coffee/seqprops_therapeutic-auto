#############################################################################
##  This script performs Friedman and Wilcoxon signed-ranks tests for all  ##
##  metrics on a specified dataset.                                        ##
#############################################################################

import argparse
from os.path import splitext
from helper_functions import get_crossvalidation_aucs, get_crossvalidation_f1, get_crossvalidation_mcc, get_crossvalidation_gm, get_crossvalidation_recall, get_crossvalidation_precision
from scipy.stats import friedmanchisquare, wilcoxon

# Parse arguments
parser = argparse.ArgumentParser(description='Plot ROC curves for all four models on a specified dataset.')
parser.add_argument('dataset_csv', type=str, help='Filename of CSV file from datasets directory.')
args = parser.parse_args()
dataset_name = splitext(args.dataset_csv)[0]

# We performed 2 times 10-fold cross-validation which resulted in 20 runs
NB_RUNS = 20

# We use significance level of 0.01 which is then corrected by Bonferroni correction.
SIGNIFICANCE_LEVEL = 0.01/3

print(
    """Notice: p-values that are higher than the corrected significance 
       level ({}) are marked with an asterisk. In those cases, no statistically 
       significant difference was found between a given representation and
       sequential properties.""".format(SIGNIFICANCE_LEVEL)
)

# Returns pvalue with asterisk if pvalue is higher than SIGNIFICANCE_LEVEL, otherwise
# returns just a pvalue.
def printable_pvalue(pvalue):
    if pvalue <= SIGNIFICANCE_LEVEL:
        return "{}".format(pvalue)
    else:
        return "{}*".format(pvalue)


# Test F1 score for significance
print("--F1---------------------------------------------------------------------------")
data = [
    get_crossvalidation_f1(dataset_name, "peptide_properties", nb_runs = NB_RUNS),
    get_crossvalidation_f1(dataset_name, "one_hot", nb_runs = NB_RUNS),
    get_crossvalidation_f1(dataset_name, "embedding", nb_runs = NB_RUNS),
    get_crossvalidation_f1(dataset_name, "sequential_properties", nb_runs = NB_RUNS),
]

friedman = friedmanchisquare(*data)
print("Friedman: {}".format(friedman.pvalue))

w = (
    wilcoxon(data[0], data[3]).pvalue,
    wilcoxon(data[1], data[3]).pvalue,
    wilcoxon(data[2], data[3]).pvalue,
    )
print("Wilcoxon (Peptide properties & Sequential properties): {}".format(printable_pvalue(w[0])))
print("Wilcoxon (One-hot & Sequential properties): {}".format(printable_pvalue(w[1])))
print("Wilcoxon (Embedding & Sequential properties): {}".format(printable_pvalue(w[2])))
print()
print()
print()



# Test MCC score for significance
print("--MCC--------------------------------------------------------------------------")
data = [
    get_crossvalidation_mcc(dataset_name, "peptide_properties", nb_runs = NB_RUNS),
    get_crossvalidation_mcc(dataset_name, "one_hot", nb_runs = NB_RUNS),
    get_crossvalidation_mcc(dataset_name, "embedding", nb_runs = NB_RUNS),
    get_crossvalidation_mcc(dataset_name, "sequential_properties", nb_runs = NB_RUNS),
]

friedman = friedmanchisquare(*data)
print("Friedman: {}".format(friedman.pvalue))

w = (
    wilcoxon(data[0], data[3]).pvalue,
    wilcoxon(data[1], data[3]).pvalue,
    wilcoxon(data[2], data[3]).pvalue,
    )
print("Wilcoxon (Peptide properties & Sequential properties): {}".format(printable_pvalue(w[0])))
print("Wilcoxon (One-hot & Sequential properties): {}".format(printable_pvalue(w[1])))
print("Wilcoxon (Embedding & Sequential properties): {}".format(printable_pvalue(w[2])))
print()
print()
print()



# Test GM score for significance
print("--GM---------------------------------------------------------------------------")
data = [
    get_crossvalidation_gm(dataset_name, "peptide_properties", nb_runs = NB_RUNS),
    get_crossvalidation_gm(dataset_name, "one_hot", nb_runs = NB_RUNS),
    get_crossvalidation_gm(dataset_name, "embedding", nb_runs = NB_RUNS),
    get_crossvalidation_gm(dataset_name, "sequential_properties", nb_runs = NB_RUNS),
]

friedman = friedmanchisquare(*data)
print("Friedman: {}".format(friedman.pvalue))

w = (
    wilcoxon(data[0], data[3]).pvalue,
    wilcoxon(data[1], data[3]).pvalue,
    wilcoxon(data[2], data[3]).pvalue,
    )
print("Wilcoxon (Peptide properties & Sequential properties): {}".format(printable_pvalue(w[0])))
print("Wilcoxon (One-hot & Sequential properties): {}".format(printable_pvalue(w[1])))
print("Wilcoxon (Embedding & Sequential properties): {}".format(printable_pvalue(w[2])))
print()
print()
print()



# Test Recall score for significance
print("--Recall-----------------------------------------------------------------------")
data = [
    get_crossvalidation_recall(dataset_name, "peptide_properties", nb_runs = NB_RUNS),
    get_crossvalidation_recall(dataset_name, "one_hot", nb_runs = NB_RUNS),
    get_crossvalidation_recall(dataset_name, "embedding", nb_runs = NB_RUNS),
    get_crossvalidation_recall(dataset_name, "sequential_properties", nb_runs = NB_RUNS),
]

friedman = friedmanchisquare(*data)
print("Friedman: {}".format(friedman.pvalue))

w = (
    wilcoxon(data[0], data[3]).pvalue,
    wilcoxon(data[1], data[3]).pvalue,
    wilcoxon(data[2], data[3]).pvalue,
    )
print("Wilcoxon (Peptide properties & Sequential properties): {}".format(printable_pvalue(w[0])))
print("Wilcoxon (One-hot & Sequential properties): {}".format(printable_pvalue(w[1])))
print("Wilcoxon (Embedding & Sequential properties): {}".format(printable_pvalue(w[2])))
print()
print()
print()



# Test Precision score for significance
print("--Precision--------------------------------------------------------------------")
data = [
    get_crossvalidation_precision(dataset_name, "peptide_properties", nb_runs = NB_RUNS),
    get_crossvalidation_precision(dataset_name, "one_hot", nb_runs = NB_RUNS),
    get_crossvalidation_precision(dataset_name, "embedding", nb_runs = NB_RUNS),
    get_crossvalidation_precision(dataset_name, "sequential_properties", nb_runs = NB_RUNS),
]

friedman = friedmanchisquare(*data)
print("Friedman: {}".format(friedman.pvalue))

w = (
    wilcoxon(data[0], data[3]).pvalue,
    wilcoxon(data[1], data[3]).pvalue,
    wilcoxon(data[2], data[3]).pvalue,
    )
print("Wilcoxon (Peptide properties & Sequential properties): {}".format(printable_pvalue(w[0])))
print("Wilcoxon (One-hot & Sequential properties): {}".format(printable_pvalue(w[1])))
print("Wilcoxon (Embedding & Sequential properties): {}".format(printable_pvalue(w[2])))
print()
print()
print()



# Test ROC-AUC for significance
print("--ROC-AUC----------------------------------------------------------------------")
data = [
    get_crossvalidation_aucs(dataset_name, "peptide_properties", nb_runs = NB_RUNS),
    get_crossvalidation_aucs(dataset_name, "one_hot", nb_runs = NB_RUNS),
    get_crossvalidation_aucs(dataset_name, "embedding", nb_runs = NB_RUNS),
    get_crossvalidation_aucs(dataset_name, "sequential_properties", nb_runs = NB_RUNS),
]

friedman = friedmanchisquare(*data)
print("Friedman: {}".format(friedman.pvalue))



w = (
    wilcoxon(data[0], data[3]).pvalue,
    wilcoxon(data[1], data[3]).pvalue,
    wilcoxon(data[2], data[3]).pvalue,
    )
print("Wilcoxon (Peptide properties & Sequential properties): {}".format(printable_pvalue(w[0])))
print("Wilcoxon (One-hot & Sequential properties): {}".format(printable_pvalue(w[1])))
print("Wilcoxon (Embedding & Sequential properties): {}".format(printable_pvalue(w[2])))