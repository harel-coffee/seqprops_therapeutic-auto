################################################################################
##  This script performs grid search, feature selection and model evaluation  ##
################################################################################

# Force execution on the CPU because of the bug in CUDA implementation on the windows
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    import argparse
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import roc_auc_score
    import models
    import peptide_encoders
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle
    from helper_functions import grid_search
    import dask
    from time import sleep
    from dask_mpi import initialize
    from dask.distributed import Client
    from seqprops import SequentialPropertiesEncoder


    parser = argparse.ArgumentParser(description='Train a model on a specified dataset using word embedding.')
    parser.add_argument('representation', type=str, choices=["peptide_properties", "one_hot", "embedding", "sequential_properties"], help='Encoding to use.')
    parser.add_argument('dataset_csv', type=str, help='Filename of CSV file from datasets directory that you would like to use for training.')
    parser.add_argument('output_prefix', type=str, help='Prefix that will be used for output files.')
    args = parser.parse_args()

    # Dask config
    dask.config.set({"distributed.comm.timeouts.tcp": "900s"})

    # Initialize DASK-MPI backend
    initialize(local_directory="/tmp")

    # Connect this local process to remote workers
    dask_client = Client()

    # Wait for workers to start
    sleep(10)
    nb_available_workers = len(dask_client.scheduler_info()["workers"])
    print("Available workers: {}".format(nb_available_workers))

    # Training parameters
    VALIDATION_SPLIT = 0.1
    N_EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    LOSS = 'binary_crossentropy'
    VERBOSE = 0

    # Load sequences from a CSV file
    data = pd.read_csv("datasets/{}".format(args.dataset_csv))

    sequences = data["sequence"].to_numpy()
    y = data["label"].to_numpy()

    # Determine the length of the longest sequence
    max_seq_len = 0
    for sequence in sequences:
        max_seq_len = max(len(sequence), max_seq_len)

    # Shuffle data
    sequences, y = shuffle(sequences, y, random_state=42)

    all_scores = []
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(sequences, y)):
        sequences_train, y_train = sequences[train_idx], y[train_idx]
        sequences_test, y_test = sequences[test_idx], y[test_idx]

        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        adam_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        if args.representation == "peptide_properties":
            create_model_fn = models.create_mlp_model
            encoder = peptide_encoders.PeptidePropertiesEncoder()
            X_train = encoder.encode(sequences_train)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            param_grid = dict(
                dense1_units=[40, 80, 120, 160],
                dense2_units=[0, 40, 80, 120, 160],
                dense3_units=[0, 40, 80, 120, 160],
            )
        elif args.representation == "one_hot":
            create_model_fn = models.create_seq_model
            encoder = peptide_encoders.OneHotEncoder(max_seq_len=max_seq_len, stop_signal=True)
            X_train = encoder.encode(sequences_train)
            param_grid = dict(
                conv1_filters=[0, 16, 32, 64],
                conv2_filters=[0, 16, 32, 64],
                conv_kernel_size=[4, 6, 8],
                num_cells=[64, 128, 256],
                dropout=[0.1, 0.2, 0.3]
            )
        elif args.representation == "embedding":
            create_model_fn = models.create_embedding_model
            encoder = peptide_encoders.IntegerTokensEncoder(max_seq_len=max_seq_len, stop_signal=True)
            X_train = encoder.encode(sequences_train)
            param_grid = dict(
                embedding_size=[30, 50, 70, 90, 110, 130],
                num_cells=[64, 128, 256],
                dropout=[0.1, 0.2, 0.3]
            )
        elif args.representation == "sequential_properties":
            create_model_fn = models.create_seq_model
            encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)), max_seq_len=max_seq_len, stop_signal=True)

            # Feature selection
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

            encoder.feature_selection(
                train_predict_fn=seqprops_train_and_predict, 
                sequences=sequences_train, 
                y=y_train, 
                nb_features="auto",
                autostop_patience=3,
                scoring='roc_auc', 
                cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=45), 
                dask_client=dask_client
            )

            # Save selected features to file
            with open("results/{}-seqprops_selected_features_{}.txt".format(args.output_prefix, split_idx), "w") as output_file:
                output_file.write(str(encoder.get_selected_properties()))

            X_train = encoder.encode(sequences_train)
            param_grid = dict(
                conv1_filters=[0, 16, 32, 64],
                conv2_filters=[0, 16, 32, 64],
                conv_kernel_size=[4, 6, 8],
                num_cells=[64, 128, 256],
                dropout=[0.1, 0.2, 0.3]
            )

        # Grid search
        best_score, best_params = grid_search(
            X_train,
            y_train,
            create_model_fn=create_model_fn,
            param_grid=param_grid,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=50),
            learning_rate=LEARNING_RATE,
            loss=LOSS,
            validation_split=VALIDATION_SPLIT,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            dask_client=dask_client
        )

        print('')
        print("Grid search results")
        print("Best hyperparameters: {}".format(best_params))
        print("ROC-AUC score: {}".format(best_score))

        # Save best parameters to file
        with open("results/{}_params_{}.txt".format(args.output_prefix, split_idx), "w") as output_file:
            output_file.write(str(best_params))

        # Instantiate model with optimal hyperparameters and train it
        if args.representation == "peptide_properties":
            X_train = encoder.encode(sequences_train)
            X_test = encoder.encode(sequences_test)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = models.create_mlp_model(
                input_shape=X_train.shape[1:],
                **best_params
            )
        elif args.representation == "one_hot":
            X_test = encoder.encode(sequences_test)
            model = models.create_seq_model(
                input_shape=X_train.shape[1:],
                **best_params
            )
        elif args.representation == "embedding":
            X_train = encoder.encode(sequences_train)
            X_test = encoder.encode(sequences_test)
            model = models.create_embedding_model(
                input_shape=X_train.shape[1:],
                **best_params
            )
        elif args.representation == "sequential_properties":
            X_train = encoder.encode(sequences_train)
            X_test = encoder.encode(sequences_test)
            model = models.create_seq_model(
                input_shape=X_train.shape[1:],
                **best_params
            )

        adam_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.compile(loss=LOSS, optimizer=adam_optimizer)
        history = model.fit(X_train, y_train, validation_split=VALIDATION_SPLIT, epochs=N_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping_callback], verbose=VERBOSE)

        y_pred = model.predict(X_test)
        print("{}  Score: {}".format(args.output_prefix, roc_auc_score(y_test, y_pred)))
        all_scores.append(roc_auc_score(y_test, y_pred))

        keras.backend.clear_session()

        # Save predictions to the file
        output_file = open("outputs/{}_{}.csv".format(args.output_prefix, split_idx), "w")
        for i in range(len(y_pred)):
            output_file.write("{}, {}\n".format(y_test[i], y_pred[i][0]))
        output_file.close()
    print("Average: {}".format(np.mean(all_scores)))
