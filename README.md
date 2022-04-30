## Sequential properties representation scheme for recurrent neural network based prediction of therapeutic peptides
**Erik Otović, Marko Njirjak, Daniela Kalafatović, Goran Mauša**

This repository contains Python scripts necessary to reproduce the results from the paper "Sequential properties representation scheme for recurrent neural network based prediction of therapeutic peptides". It also contains the data and the results we obtained.


### Steps to reproduce the results

1. Start by cloning this repository to your local computer. Once cloned, open the terminal and navigate to the root directory of the cloned repository.

2. Create new virtual environment with Python 3.9.4 and install required packages from the *environment.yml* file:

    > pip install -r environment.yml


3. Four datasets that we used in our experiment are present in *datasets* directory and we use them to train the models that are based on different peptide representations. Let's start by analyzing the content of the datasets.

    3.1. Plot the distribution of peptide lengths

    > python plot_peptide_lengths.py avpdb.csv

    > python plot_peptide_lengths.py avdramp.csv

    > python plot_peptide_lengths.py avmerged.csv

    > python plot_peptide_lengths.py amp.csv

    3.2. Plot the distribution of similarities for all peptide pairs in a dataset

    > python plot_similarities.py avpdb.csv

    > python plot_similarities.py avdramp.csv

    > python plot_similarities.py avmerged.csv

    > python plot_similarities.py amp.csv

    Notice:<em>This takes about 1.5 days for the largest dataset (amp.csv) on a 24 core machine. The reason for this is the slow python implementation of sequence alignment algorithm. Similarity plots will be saved in results directory.</em>

 4. In our experiment, we use 2 times 10-fold stratified cross validation. The way how datasets are split into training and test sets is the same for all models. Therefore, all models are trained and evaluated on the same train/test splits and in the same order so they can be fairly compared. Training script will at each repetition save ground truth labels along with the predicted probabilities for test set in *outputs* directory. Saved files have the filename in the following format: &lt;representation name&gt;_&lt;dataset name&gt;_&lt;repetition number&gt;.csv

    Scripts are intended to be executed on MPI cluster, so you can modify the number of MPI processes in the following commands according to your computational resources.

    4.1. Train model based on peptide properties representation

    > mpirun -np 540 python train.py peptide_properties avpdb.csv peptide_properties_avpdb

    > mpirun -np 540 python train.py peptide_properties avdramp.csv peptide_properties_avdramp

    > mpirun -np 540 python train.py peptide_properties avmerged.csv peptide_properties_avmerged

    > mpirun -np 540 python train.py peptide_properties amp.csv peptide_properties_amp


    4.2. Train model based on one-hot vector encoding representation

    > mpirun -np 540 python train.py one_hot avpdb.csv one_hot_avpdb

    > mpirun -np 540 python train.py one_hot avdramp.csv one_hot_avdramp

    > mpirun -np 540 python train.py one_hot avmerged.csv one_hot_avmerged

    > mpirun -np 540 python train.py one_hot amp.csv one_hot_amp


    4.3. Train model based on word embedding representation

    > mpirun -np 540 python train.py embedding avpdb.csv embedding_avpdb

    > mpirun -np 540 python train.py embedding avdramp.csv embedding_avdramp

    > mpirun -np 540 python train.py embedding avmerged.csv embedding_avmerged

    > mpirun -np 540 python train.py embedding amp.csv embedding_amp


    4.4. Train model based on sequential properties representation

    > mpirun -np 540 python train.py sequential_properties avpdb.csv sequential_properties_avpdb

    > mpirun -np 540 python train.py sequential_properties avdramp.csv sequential_properties_avdramp

    > mpirun -np 540 python train.py sequential_properties avmerged.csv sequential_properties_avmerged

    > mpirun -np 540 python train.py sequential_properties amp.csv sequential_properties_amp


5. Execute *plot_roc.py* to plot ROC curve and calculate ROC-AUC on a given dataset.

    > python plot_roc.py avpdb.csv

    > python plot_roc.py avdramp.csv

    > python plot_roc.py avmerged.csv

    > python plot_roc.py amp.csv


6. Execute *plot_f1_boxplots.py* to plot distribution of F1 score in the form of box plots on a given dataset. This script can be easily modified to display the distribution of any other score.

    > python plot_f1_boxplots.py avpdb.csv

    > python plot_f1_boxplots.py avdramp.csv

    > python plot_f1_boxplots.py avmerged.csv

    > python plot_f1_boxplots.py amp.csv


7. Execute *plot_f1_classification_threshold.py* to plot how F1 score changes with the respect to the chosen classification threshold.

    > python plot_f1_classification_threshold.py avpdb.csv

    > python plot_f1_classification_threshold.py avdramp.csv

    > python plot_f1_classification_threshold.py avmerged.csv

    > python plot_f1_classification_threshold.py amp.csv

8. Execute *print_scores.py* to compute F1 score, Matthews correlation coefficient, geometric mean score, recall and precision for all four models on a given dataset. Note that ROC-AUC is computed by *plot_roc.py* script.

    > python print_scores.py avpdb.csv

    > python print_scores.py avdramp.csv

    > python print_scores.py avmerged.csv

    > python print_scores.py amp.csv

9. Execute *statistical_test.py* to conduct statistical tests on a given dataset. For each metric a seperate Friedman test is performed with significance level of 0.01. Wilcoxon-signed ranks test is used in post-hoc analysis to compare sequential properties against other three representations. The significance level used for post-hoc analysis is 0.003 which is obtained by adjusting the significance level of 0.01 with Bonferroni correction. 

    > python statistical_test.py avpdb.csv

    > python statistical_test.py avdramp.csv

    > python statistical_test.py avmerged.csv

    > python statistical_test.py amp.csv

10. Analyse the dependency of ROC-AUC on the number of selected features for <em>sequential properties</em> encoding scheme.

    > mpirun -np 540 python seqprops_rank_features.py avpdb.csv

    > mpirun -np 540 python seqprops_rank_features.py avdramp.csv

    > mpirun -np 540 python seqprops_rank_features.py avmerged.csv

    > mpirun -np 540 python seqprops_rank_features.py amp.csv

    Notice: <em>This can take about one day to execute for the largest dataset (amp.csv).</em>

### Prediction script
Prediction script with pretrained models is available in this repository. You must clone the repository as was described in steps 1. and 2. before you can use prediction script. There are four pretrained models that are available through prediction script and they are named after the dataset they were trained on (avpdb, avdramp, avmerged, amp). For example, to predict the probabilities of sequence ACA and CAC having antiviral properties, one can use one of the following commands:

> python prediction_script.py avpdb ACA CAC

> python prediction_script.py avdramp ACA CAC

> python prediction_script.py avmerged ACA CAC

The same can be done for antimicrobial activity:

> python prediction_script.py amp ACA CAC

### Generating reports from our data

The *our_outputs* directory contains predicted probabilities that we obtained and used to generate tables and plots presented in the paper. You can use this data to generate reports without conducting your own training. To do this, simply copy the content of *our_outputs* directory into the *outputs* directory and run any command from the 5th, 6th, 7th, 8th or 9th step of the previous section.

The reports we generated that were published in our paper can be found inside the *[our_reports](our_reports)* directory.



### Standalone <em>seqprops</em> package

If you would like to use <em>sequential properties</em> scheme in your work, we refer you to our [Python package](https://pypi.org/project/seqprops/). 

---

Execute any python script with *-h* flag to check for available options.

*Feel free to contact me at <erik.otovic@gmail.com>*
