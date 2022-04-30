#######################################################
##  This script contains models used in experiment.  ##
#######################################################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D

def create_mlp_model(input_shape, dense1_units=100, dense2_units=100, dense3_units=100):
    model_input = Input(shape=input_shape)
    x = Dense(dense1_units, activation='relu')(model_input)

    if dense2_units > 0:
        x = Dense(dense2_units, activation='relu')(x)

    if dense3_units > 0:
        x = Dense(dense3_units, activation='relu')(x)

    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=model_input, outputs=x)
    return model

def create_embedding_model(input_shape, embedding_size=128, num_cells=64, dropout=0.2):
    model_input = Input(shape=input_shape)
    x = Embedding(21, embedding_size, mask_zero=True)(model_input)
    x = Bidirectional(LSTM(num_cells, unroll=True))(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=model_input, outputs=x)
    return model

def create_seq_model(input_shape, conv1_filters=64, conv2_filters=64, conv_kernel_size=6, num_cells=64, dropout=0.1):
    model_input = Input(shape=input_shape, name="input_1")
    if conv1_filters > 0:
        x = Conv1D(conv1_filters, conv_kernel_size, padding='same', kernel_initializer='he_normal', name="conv1d_1")(model_input)
        
        if conv2_filters > 0:
            x = Conv1D(conv2_filters, conv_kernel_size, padding='same', kernel_initializer='he_normal', name="conv1d_2")(x)
        x = Bidirectional(LSTM(num_cells, unroll=True, name="bi_lstm"))(x)
    else:
        x = Bidirectional(LSTM(num_cells, unroll=True,  name="bi_lstm"))(model_input)

    if dropout > 0:
        x = Dropout(dropout, name="dropout")(x)

    x = Dense(1, activation='sigmoid', name="output_dense")(x)
    model = Model(inputs=model_input, outputs=x)
    return model