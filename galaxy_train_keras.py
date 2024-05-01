import os, h5py
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils


class DataGenerator(keras.utils.Sequence):

    def __init__(self, x, y, val=False, batch_size=16):
        self.x          = x
        self.y          = y
        self.batch_size = batch_size
        self.val        = val
        self.indexes    = np.arange(len(self.x))
        if not self.val:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True
            )
        else:
            self.datagen = ImageDataGenerator()

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        X = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        if not self.val:
            X = self.datagen.flow(X, batch_size=self.batch_size, shuffle=False).next()
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


def get_model():

    inputs  = keras.layers.Input(shape=(69, 69, 3))
    x       = keras.applications.ResNet50V2(include_top=False, input_shape=(69, 69, 3), weights=None)(inputs)
    x       = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Dropout(0.3, name="top_dropout")(x)
    x       = keras.layers.Dense(256, activation="relu")(x)
    x       = keras.layers.BatchNormalization()(x)
    x       = keras.layers.Dense(64, activation="relu")(x)
    x       = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(10, name="pred", activation="softmax")(x)
    model   = keras.Model(inputs, outputs, name="ResNet50V2")

    return model

def get_train_val(data_path):
    with h5py.File(data_path, 'r') as file:
        data       = file['images'][10:].astype(np.float32)
        labels     = file['ans'][10:].astype(np.float32)
    labels = utils.to_categorical(labels, 10)
    train_x, val_x, train_y, val_y = train_test_split(data, labels, test_size=0.2)
    return train_x, train_y, val_x, val_y


if __name__ == '__main__':

    train_x, train_y, val_x, val_y = get_train_val(data_path='./Data/Galaxy10.h5')
    batch_size                     = 64
    one_epoch_steps                = len(train_x) / batch_size
    train_data                     = DataGenerator(x=train_x, y=train_y, batch_size=batch_size)
    val_data                       = DataGenerator(x=val_x, y=val_y, val=True, batch_size=batch_size)

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir                = 'logs_keras/'
        ),
        keras.callbacks.ModelCheckpoint(
            'logs_keras/Epoch-{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor                = 'val_loss',
            save_weights_only      = False,
            save_best_only         = False,
            save_freq              = 'epoch'
        )
    ]

    lr_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate      = 1e-2,
        decay_steps                = 6000,
        alpha                      = 1e-3,
    )

    optimizer                      = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) # (learning_rate=lr_decay, epsilon=0.1)

    if True:
        model                      = get_model()
        model.compile(
            optimizer              = optimizer,
            loss                   = keras.losses.CategoricalCrossentropy(label_smoothing=1e-3),
            metrics                = ["accuracy"]
        )
        model.fit(
            x                      = train_data,
            validation_data        = val_data,
            steps_per_epoch        = len(train_x) // batch_size,
            validation_steps       = len(val_x) // batch_size,
            initial_epoch          = 0,
            epochs                 = 50,
            callbacks              = callbacks,
            use_multiprocessing    = False,
            workers                = 0
        )
