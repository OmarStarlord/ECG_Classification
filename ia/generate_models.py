#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
from tensorflow.keras import layers, models, regularizers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_ecg200_data():
    """
    Load the real ECG200 dataset from the original URLs
    
    :return: tuple of (X_train_std, y_train, X_test_std, y_test)
    """
    TRAIN_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv"
    TEST_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv"
    
    logging.info("Loading ECG200 dataset...")
    
    # Load data
    ecg200_train_df = pd.read_csv(TRAIN_URL, sep='\t', header=None)
    ecg200_test_df = pd.read_csv(TEST_URL, sep='\t', header=None)
    
    # Separate features and labels
    y_train = ecg200_train_df.iloc[:, 0]
    x_train = ecg200_train_df.iloc[:, 1:]
    
    y_test = ecg200_test_df.iloc[:, 0]
    x_test = ecg200_test_df.iloc[:, 1:]
    
    # Standardize features
    x_train_std = (x_train - x_train.mean()) / x_train.std()
    x_test_std = (x_test - x_train.mean()) / x_train.std()
    
    # One-hot encode labels
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    logging.info(f"Training data shape: {x_train_std.shape}")
    logging.info(f"Test data shape: {x_test_std.shape}")
    
    return x_train_std, y_train, x_test_std, y_test

def create_mlp_model():
    """
    Create MLP model using exact architecture from original file
    
    :return: compiled MLP model
    """
    input_size = 96
    hidden_states_sizes = [30, 20]
    output_size = 2
    
    # init model
    model = models.Sequential(name='MLP')
    
    # input
    model.add(layers.Input(shape=(input_size,)))
    
    # deep layers
    for idx, h in enumerate(hidden_states_sizes):
        model.add(layers.Dense(h, activation='relu', name='Dense_'+str(idx), 
                              kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.3))
    
    # output
    model.add(layers.Dense(output_size, activation='softmax', name='Output'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn_model():
    """
    Create CNN model using exact architecture from original file
    
    :return: compiled CNN model
    """
    input_shape = (96, 1)
    input_layer = tf.keras.layers.Input(input_shape)
    
    hidden_conv_layer_1 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(input_layer)
    pooling_1 = tf.keras.layers.MaxPooling1D(pool_size=2)(hidden_conv_layer_1)
    dropout_1 = tf.keras.layers.Dropout(0.3)(pooling_1)
    
    hidden_conv_layer_2 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(dropout_1)
    pooling_2 = tf.keras.layers.MaxPooling1D(pool_size=2)(hidden_conv_layer_2)
    dropout_2 = tf.keras.layers.Dropout(0.4)(pooling_2)
    
    flatten = tf.keras.layers.Flatten()(dropout_2)
    output = tf.keras.layers.Dense(2, activation='softmax')(flatten)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def create_rnn_model():
    """
    Create RNN model using exact architecture from original file
    
    :return: compiled RNN model
    """
    input_shape = (96, 1)
    input_layer = tf.keras.layers.Input(input_shape)
    
    lstm_1 = layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2)(input_layer)
    dropout_1 = layers.Dropout(0.3)(lstm_1)
    
    lstm_2 = layers.LSTM(32, return_sequences=False, recurrent_dropout=0.2)(dropout_1)
    dropout_2 = layers.Dropout(0.3)(lstm_2)
    
    output = layers.Dense(2, activation='softmax')(dropout_2)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

def augment_ecg(X, y, noise_level=0.01, shift_max=5):
    """
    Data augmentation for ECG signals (from original file)
    
    :param X: input data
    :param y: labels
    :param noise_level: noise level for augmentation
    :param shift_max: maximum shift for augmentation
    :return: augmented data and labels
    """
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        signal = X[i]
        label = y[i]
        
        X_aug.append(signal)
        y_aug.append(label)
        
        # Add noise
        noisy = signal + np.random.normal(0, noise_level, signal.shape)
        X_aug.append(noisy)
        y_aug.append(label)
        
        # Add shift
        shift = np.random.randint(-shift_max, shift_max)
        shifted = np.roll(signal, shift)
        X_aug.append(shifted)
        y_aug.append(label)
    
    return np.array(X_aug), np.array(y_aug)

def train_and_save_model(model, model_name, X_train, y_train, epochs=50):
    """
    Train and save a model using the approach from original file
    
    :param model: the model to train
    :param model_name: name for saving
    :param X_train: training data
    :param y_train: training labels
    :param epochs: number of training epochs
    """
    logging.info(f"Training {model_name} model...")
    
    # Prepare data based on model type
    if model_name == 'mlp':
        X_train_prepared = X_train
        batch_size = 16
        validation_split = 0.3
        callbacks = []
    elif model_name == 'cnn':
        X_train_prepared = X_train.values.reshape(-1, 96, 1)
        batch_size = 16
        epochs = 100
        validation_split = 0.3
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    elif model_name == 'rnn':
        X_train_prepared = X_train.values.reshape(-1, 96, 1)
        # Apply data augmentation for RNN
        X_train_prepared, y_train = augment_ecg(X_train_prepared, y_train.values)
        batch_size = 32
        epochs = 100
        validation_split = 0.2
        class_weight = {0: 1.0, 1: 2.23}
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        ]
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Train the model
    history = model.fit(
        X_train_prepared, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1 if model_name != 'rnn' else 1,
        **({'class_weight': class_weight} if model_name == 'rnn' else {})
    )
    
    # Save the model
    model_path = f"./graph/{model_name}_model.h5"
    os.makedirs('./graph', exist_ok=True)
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    return model

def generate_all_models():
    """
    Generate all three models (MLP, CNN, RNN) using real ECG200 data
    """
    logging.info("Starting model generation with real ECG200 data...")
    
    # Load real training data
    X_train, y_train, X_test, y_test = load_ecg200_data()
    
    # Create and train models
    models = {
        'mlp': create_mlp_model(),
        'cnn': create_cnn_model(),
        'rnn': create_rnn_model()
    }
    
    for model_name, model in models.items():
        try:
            train_and_save_model(model, model_name, X_train, y_train)
            logging.info(f"Successfully generated {model_name} model")
        except Exception as e:
            logging.error(f"Failed to generate {model_name} model: {e}")
    
    logging.info("Model generation complete!")

if __name__ == "__main__":
    generate_all_models()
