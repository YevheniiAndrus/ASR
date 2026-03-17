import pathlib, os, configparser
import librosa
import random

import numpy as np
import re, hashlib

import tensorflow as tf
import keras
from keras import layers
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# utility function to determine dataset for file
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
def which_set(filename, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                    (MAX_NUM_WAVS_PER_CLASS + 1)) *
                    (100.0 / MAX_NUM_WAVS_PER_CLASS))
        
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    
    return result

def build_mel_spectrogram_train(path):
    def preprocess(path_str):
        utf8_path = path_str.numpy().decode('utf-8')
        speech, sampling_rate = librosa.load(utf8_path)
        
        # data augmentation
        random_bool = bool(random.getrandbits(1)) # choose if add some noise or not
        if random_bool == True:
            noise_path = os.path.join(background_noise_dir, random.choice(noise_list))
            
            noise, _ = librosa.load(noise_path)
            start = np.random.randint(0, max(1, len(noise) - len(speech)))
            noise = noise[start:start+len(speech)]
            
            # signal to noise ration
            snr_db = random.randint(1, 10)
            speech_power = np.mean(speech**2)
            noise_power = np.mean(noise**2)

            scaling_factor = np.sqrt(speech_power / (10**(snr_db / 10) * noise_power))
            noise_scaled = noise * scaling_factor

            speech = speech + noise_scaled
        
        S = librosa.feature.melspectrogram(y=speech, sr=sampling_rate, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # make all S_dB the same shape (44, 128)
        S_fixed = librosa.util.fix_length(S_dB, size=44, axis=1)
        S_fixed = S_fixed.T

        # find label
        class_name = pathlib.Path(utf8_path).parent.name
        label = classes.index(class_name)

        return S_fixed, label
    
    # Wrap Python function and return tensors
    spectrogram, label = tf.py_function(
        func=preprocess,
        inp=[path],
        Tout=(tf.float32, tf.int32)
    )

    # Set static shapes so tf.data knows dimensions
    spectrogram.set_shape((44, 128))
    label.set_shape(())

    return spectrogram, label

def build_mel_spectrogram_val_test(path):
    def preprocess(path_str):
        utf8_path = path_str.numpy().decode('utf-8')

        speech, sampling_rate = librosa.load(utf8_path)
        
        S = librosa.feature.melspectrogram(y=speech, sr=sampling_rate, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # make all S_dB the same shape (44, 128)
        S_fixed = librosa.util.fix_length(S_dB, size=44, axis=1)
        S_fixed = S_fixed.T

        # find label
        class_name = pathlib.Path(utf8_path).parent.name
        label = classes.index(class_name)

        return S_fixed, label
    
    # Wrap Python function and return tensors
    spectrogram, label = tf.py_function(
        func=preprocess,
        inp=[path],
        Tout=(tf.float32, tf.int32)
    )

    # Set static shapes so tf.data knows dimensions
    spectrogram.set_shape((44, 128))
    label.set_shape(())

    return spectrogram, label

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config_file_path = 'config.cfg'

    # read configuration
    config.read(config_file_path)

    google_speech_dataset_path = pathlib.Path(config.get("dataset", "dataset_path"))
    validation_percentage      = config.getfloat("dataset", "valid_size")
    testing_percentage         = config.getfloat("dataset", "test_size")

    lstm_size    = config.getint("model", "lstm_size")
    dense_size   = config.getint("model", "dense_size")
    dropout_rate = config.getfloat("model", "dropout")

    num_epochs   = config.getint("training", "num_epochs")

    background_noise_dir = google_speech_dataset_path / "_background_noise_"
    noise_list = [noise for noise in os.listdir(background_noise_dir) if not noise.endswith('.md')]

    # every folder is a class of command that contains 1 second *.wav files
    classes = os.listdir(google_speech_dataset_path)
    classes = [cls for cls in classes if os.path.isdir(google_speech_dataset_path / cls)
          and not cls.startswith("_")]
    
    # gather all the data into a single array
    data = []
    for cls in classes:
        files = os.listdir(google_speech_dataset_path / cls)
        data.extend([str(google_speech_dataset_path / cls / file) for file in files])

    random.shuffle(data)

    # split dataset into training, validation and test
    training_data = []
    validation_data = []
    test_data = []

    for path in data:
        result = which_set(str(path), validation_percentage, testing_percentage)
        
        if result == 'training':
            training_data.append(str(path))
        elif result == 'validation':
            validation_data.append(str(path))
        else:
            test_data.append(str(path))

    # Create train and validation datasets
    training_dataset = tf.data.Dataset.from_tensor_slices(training_data)
    training_dataset = training_dataset.map(build_mel_spectrogram_train, num_parallel_calls=8).batch(32).prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
    validation_dataset = validation_dataset.map(build_mel_spectrogram_val_test, num_parallel_calls=8).batch(32).prefetch(tf.data.AUTOTUNE)

    # define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="asr.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # build model
    inputs = keras.Input(shape=(44, 128))
    x = layers.Normalization()(inputs)

    x = layers.LSTM(lstm_size, return_sequences=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.LSTM(lstm_size)(x)
    x = layers.Dropout(dropout_rate)(x)

    # classification
    x = layers.Dense(dense_size, activation="relu")(x)
    x = layers.Dropout(dropout_rate / 3)(x)

    outputs = layers.Dense(len(classes), activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    #model.summary()
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=5e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    history = model.fit(training_dataset,
            validation_data=validation_dataset,
            callbacks=callbacks,
            epochs=num_epochs)
    
    # Plot loss and accuracy
    # Loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # build test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_dataset = test_dataset.map(build_mel_spectrogram_val_test, num_parallel_calls=8).batch(32).prefetch(tf.data.AUTOTUNE)

    # calculate test accuracy
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy - {test_acc:.4f}")