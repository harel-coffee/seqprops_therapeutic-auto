#####################################################################################
##  This script contains some helper functions that are invoked in other scripts.  ##
#####################################################################################

from sklearn import metrics
import numpy as np
import itertools
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, precision_score
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# This function loads predicted probabilities for a given method on a given dataset.
# All predicted probabilities on test sets of nb_runs splits are concatenated and returned.
def load_predicted_probabilities(dataset_name, method, nb_runs):
    probabilities = []
    for i in range(nb_runs):
        y_pred = []
        file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, i)
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        
        for line in lines:
            y_pred.append(float(line.split(",")[1]))
        probabilities.append(y_pred)
    return probabilities

# This function loads true labels for a given method on a given dataset.
# All true labels on test sets of nb_runs splits are concatenated and returned.
def load_true_labels(dataset_name, method, nb_runs):
    labels = []
    for i in range(nb_runs):
        y_true = []
        # Each output CSV file contains predicted and ground truth values
        file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, i)
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        
        for line in lines:
            y_true.append(int(line.split(",")[0]))
        labels.append(y_true)
    return labels

def compute_avg_auc(dataset_name, method, nb_runs):
    labels = load_true_labels(dataset_name, method, nb_runs)
    probabilities = load_predicted_probabilities(dataset_name, method, nb_runs)
    aucs = []
    for y_true, y_pred in zip(labels, probabilities):
        auc = metrics.roc_auc_score(y_true, y_pred)
        aucs.append(auc)
    return np.mean(aucs)

# Computes geometric mean score that is defined as geometric mean of recall and specificity.
def geometric_mean_score(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr * tnr)**0.5
   
# Computes AUC score on each test set of nb_runs splits.
# Returns an aeeay with nb_runs values.
def get_crossvalidation_aucs(dataset_name, method, nb_runs):
    aucs = []
    for run_nb in range(int(nb_runs/10)):
        for j in range(10):
            file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, run_nb * 10 + j)
            f = open(file_path, "r")
            lines = f.readlines()
            f.close()
            
            y_true = []
            y_pred = []
            for line in lines:
                y_true.append(int(line.split(",")[0]))
                y_pred.append(float(line.split(",")[1]))
            aucs.append(metrics.roc_auc_score(y_true, y_pred))
    return aucs

# Computes F1 score on each test set of nb_runs splits.
# Returns an aeeay with nb_runs values.
def get_crossvalidation_f1(dataset_name, method, nb_runs, threshold=0.5):
    scores = []
    for i in range(nb_runs):
        file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, i)
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        
        y_true = []
        y_pred = []
        for line in lines:
            y_true.append(int(line.split(",")[0]))
            y_pred.append(float(line.split(",")[1]))
        y_pred = [1 if val > threshold else 0 for val in y_pred]
        scores.append(f1_score(y_true, y_pred))
    return scores

# Computes recall on each test set of nb_runs splits.
# Returns an aeeay with nb_runs values.
def get_crossvalidation_recall(dataset_name, method, nb_runs, threshold=0.5):
    scores = []
    for i in range(nb_runs):
        file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, i)
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        
        y_true = []
        y_pred = []
        for line in lines:
            y_true.append(int(line.split(",")[0]))
            y_pred.append(float(line.split(",")[1]))
        y_pred = [1 if val > threshold else 0 for val in y_pred]
        scores.append(recall_score(y_true, y_pred))
    return scores

# Computes precision on each test set of nb_runs splits.
# Returns an aeeay with nb_runs values.
def get_crossvalidation_precision(dataset_name, method, nb_runs, threshold=0.5):
    scores = []
    for i in range(nb_runs):
        file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, i)
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        
        y_true = []
        y_pred = []
        for line in lines:
            y_true.append(int(line.split(",")[0]))
            y_pred.append(float(line.split(",")[1]))
        y_pred = [1 if val > threshold else 0 for val in y_pred]
        scores.append(precision_score(y_true, y_pred))
    return scores

# Computes geometric mean score on each test set of nb_runs splits.
# Returns an aeeay with nb_runs values.
def get_crossvalidation_gm(dataset_name, method, nb_runs, threshold=0.5):
    scores = []
    for i in range(nb_runs):
        file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, i)
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        
        y_true = []
        y_pred = []
        for line in lines:
            y_true.append(int(line.split(",")[0]))
            y_pred.append(float(line.split(",")[1]))
        y_pred = [1 if val > threshold else 0 for val in y_pred]
        scores.append(geometric_mean_score(y_true, y_pred))
    return scores    
    
# Computes MCC on each test set of nb_runs splits.
# Returns an aeeay with nb_runs values.
def get_crossvalidation_mcc(dataset_name, method, nb_runs, threshold=0.5):
    scores = []
    for i in range(nb_runs):
        file_path = "outputs/{}_{}_{}.csv".format(method, dataset_name, i)
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        
        y_true = []
        y_pred = []
        for line in lines:
            y_true.append(int(line.split(",")[0]))
            y_pred.append(float(line.split(",")[1]))
        y_pred = [1 if val > threshold else 0 for val in y_pred]
        scores.append(matthews_corrcoef(y_true, y_pred))
    return scores

# Trains newly instantiated model and then makes a prediction.
def train_and_predict(create_model_fn, learning_rate, loss, validation_split, n_epochs, batch_size, model_params={}):
    from sklearn.model_selection import train_test_split
    def f(X_train, y_train, X_test):
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        adam_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, stratify=y_train, random_state=42)

        model = create_model_fn(input_shape=X_train.shape[1:], **model_params)
        model.compile(loss=loss, optimizer=adam_optimizer)
        model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=n_epochs, 
            batch_size=batch_size, 
            callbacks=[early_stopping_callback],
            verbose=0
        )
        y_pred = model.predict(X_test)
        del model
        tf.keras.backend.reset_uids()
        tf.keras.backend.clear_session()
        return y_pred
    return f

# Evaluates a model.
def evaluate_model(create_model_fn, model_params, X, y, train_idx, test_idx, learning_rate, loss, validation_split, n_epochs, batch_size):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_predict_fn = train_and_predict(create_model_fn, learning_rate, loss, validation_split, n_epochs, batch_size, model_params)
    y_test_pred = train_predict_fn(X_train, y_train, X_test)
    score = roc_auc_score(y_test, y_test_pred)
    return score

# Taken from https://stackoverflow.com/a/40623158
def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

# Grid search used for hyperparameter optimization.
def grid_search(X, y, create_model_fn, param_grid, cv, learning_rate, loss, validation_split, n_epochs, batch_size, dask_client):
    X_future = dask_client.scatter(X, broadcast=True)
    y_future = dask_client.scatter(y, broadcast=True)

    combinations_of_parameters = list(dict_product(param_grid))
    scores = []
    for params in combinations_of_parameters:
        cv_scores = []
        for train_idx, test_idx in cv.split(X, y):
            cv_scores.append(dask_client.submit(
                evaluate_model,
                create_model_fn,
                params,
                X_future,
                y_future,
                train_idx,
                test_idx,
                learning_rate,
                loss,
                validation_split,
                n_epochs,
                batch_size,
                pure=False
            ))
        scores.append(dask_client.submit(
            np.mean,
            cv_scores,
            pure=False
        ))

    scores = dask_client.gather(scores)
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_params = combinations_of_parameters[best_idx]
    return (best_score, best_params)

