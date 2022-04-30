###############################################################
##  This script uses pretrained models to make a prediction  ##
###############################################################

# Force execution on the CPU because of the bug in CUDA implementation on the windows
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    import argparse
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    import models
    from seqprops import SequentialPropertiesEncoder


    parser = argparse.ArgumentParser(description='This script uses pretrained model to make a prediction. The output values represent a probability that a sequence is active.')
    parser.add_argument('pretrained_model', type=str, choices=["avpdb", "avdramp", "avmerged", "amp"], help='Pretrained model to use.')
    parser.add_argument('sequences', nargs='+', type=str)
    args = parser.parse_args()

    feature_set = None
    if args.pretrained_model == "avpdb":
        feature_set = ['Hydrophobicity_Argos', 'FASGAI_F6', 'BLOSUM_BLOSUM10', 'Hydrophobicity_BullBreese', 'crucianiProperties_PP3', 'BLOSUM_BLOSUM7', 'Hydrophobicity_Manavalan', 'VHSE_VHSE8']
        encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)), stop_signal=True, selected_properties=feature_set)
        encoded_sequences = encoder.encode(args.sequences)
        model = models.create_seq_model(input_shape=encoded_sequences.shape[1:], conv1_filters=64, conv2_filters=64, conv_kernel_size=8, num_cells=64, dropout=0.1)

    elif args.pretrained_model == "avmerged":
        feature_set = ['Hydrophobicity_Fauchere', 'stScales_ST5', 'ProtFP_ProtFP7', 'Hydrophobicity_Kuhn', 'ProtFP_ProtFP6', 'BLOSUM_BLOSUM7', 'VHSE_VHSE7', 'Hydrophobicity_BullBreese', 'FASGAI_F6', 'VHSE_VHSE4', 'VHSE_VHSE8']
        encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)), stop_signal=True, selected_properties=feature_set)
        encoded_sequences = encoder.encode(args.sequences)
        model = models.create_seq_model(input_shape=encoded_sequences.shape[1:], conv1_filters=64, conv2_filters=64, conv_kernel_size=6, num_cells=64, dropout=0.2)
    
    elif args.pretrained_model == "avdramp":
        feature_set = ['Hydrophobicity_Aboderin', 'FASGAI_F4', 'Hydrophobicity_BullBreese', 'stScales_ST4', 'FASGAI_F5', 'stScales_ST8', 'VHSE_VHSE7', 'FASGAI_F3']
        encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)), stop_signal=True, selected_properties=feature_set)
        encoded_sequences = encoder.encode(args.sequences)
        model = models.create_seq_model(input_shape=encoded_sequences.shape[1:], conv1_filters=16, conv2_filters=64, conv_kernel_size=6, num_cells=128, dropout=0.3)

    elif args.pretrained_model == "amp":
        feature_set = ['Hydrophobicity_Aboderin', 'tScales_T3', 'Hydrophobicity_Fauchere', 'stScales_ST4', 'VHSE_VHSE3', 'BLOSUM_BLOSUM5', 'Hydrophobicity_Ponnuswamy', 'crucianiProperties_PP1', 'Hydrophobicity_Welling', 'Hydrophobicity_Chothia']
        encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)), stop_signal=True, selected_properties=feature_set)
        encoded_sequences = encoder.encode(args.sequences)
        model = models.create_seq_model(input_shape=encoded_sequences.shape[1:], conv1_filters=64, conv2_filters=64, conv_kernel_size=4, num_cells=128, dropout=0.1)

    # Copy weights from pretrained model to the new model
    loaded_model = tf.keras.models.load_model("pretrained_models/{}".format(args.pretrained_model))
    model.set_weights(loaded_model.get_weights())

    y_pred = model.predict(encoded_sequences)
    for idx, seq in enumerate(args.sequences):
        print("{},{}".format(seq, y_pred[idx][0]))