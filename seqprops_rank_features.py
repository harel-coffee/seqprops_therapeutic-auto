#######################################################################################
##  This script plots the dependency of ROC-AUC on the number of selected features.  ##
#######################################################################################

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import argparse
    import pandas as pd
    import models
    from seqprops import SequentialPropertiesEncoder
    from tensorflow import keras
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.utils import shuffle
    import json
    import tensorflow as tf
    import dask
    from dask_mpi import initialize
    from dask.distributed import Client
    from sklearn.model_selection import RepeatedStratifiedKFold

    # Initialize dask config
    dask.config.set({'temporary_directory': '/tmp'})
    
    # Initialize DASK-MPI backend
    initialize(local_directory="/tmp")

    # Connect this local process to remote workers
    dask_client = Client()

    parser = argparse.ArgumentParser(description="Plot the dependency of ROC-AUC on the number of selected features.")
    parser.add_argument('dataset_csv', type=str, help='Filename of CSV file from datasets directory that you would like to use for analysis.')
    args = parser.parse_args()

    # Portion of the dataset used to estimate validation loss so early stopping mechanism can be used
    VALIDATION_SPLIT = 0.1

    # Maximal number of epochs
    N_EPOCHS = 200

    # Batch size
    BATCH_SIZE = 32

    # Learning rate
    LEARNING_RATE = 0.0001

    # Loss
    LOSS = 'binary_crossentropy'

    # Number of selected features and corresponding scoress will be stored in these arrays
    x_nb_features = []
    y_cv_score = []

    # Load sequences from a CSV file
    data = pd.read_csv("datasets/{}".format(args.dataset_csv))

    sequences = data["sequence"].to_numpy()
    y = data["label"].to_numpy()

    # Shuffle training data
    sequences, y = shuffle(sequences, y, random_state=42)

    # Initialize encoder
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)), stop_signal=True)
    total_nb_features = len(encoder.get_available_properties())

    def seqprops_train_and_predict(X_train, y_train, X_test):
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        adam_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        model = models.create_seq_model(input_shape=X_train.shape[1:])
        model.compile(loss=LOSS, optimizer=adam_optimizer)
        model.fit(
            X_train, y_train, 
            validation_split=VALIDATION_SPLIT, 
            epochs=N_EPOCHS, 
            batch_size=BATCH_SIZE, 
            callbacks=[early_stopping_callback],
            verbose=0
        )
        y_pred = model.predict(X_test)
        del model
        tf.keras.backend.reset_uids()
        tf.keras.backend.clear_session()
        return y_pred


    history = encoder.feature_selection(
        train_predict_fn=seqprops_train_and_predict, 
        sequences=sequences, 
        y=y, 
        nb_features=total_nb_features, 
        scoring='roc_auc', 
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=50), 
        dask_client=dask_client
    )

    # Save results of feature selection to the json file
    filename = os.path.splitext(args.dataset_csv)[0]
    with open("results/{}-roc_auc-nb_features.json".format(filename), "w") as f:
        json_str = json.dumps(history)
        f.write(json_str)

    # Plot the dependency of ROC-AUC and number of selected features
    x = [len(selection_step['selected_features']) for selection_step in history]
    y = [selection_step['score'] for selection_step in history]
    plt.plot(x, y)

    plt.suptitle("Dependency of ROC-AUC on the number of selected features")
    plt.xlabel("Number of selected features")
    plt.ylabel("ROC-AUC")
    plt.savefig("results/{}-roc_auc-nb_features.png".format(filename))
