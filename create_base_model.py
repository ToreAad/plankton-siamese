import os
import matplotlib.pyplot as plt
import numpy as np

import pickle

from sklearn.metrics import confusion_matrix


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, \
    Concatenate, Lambda, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from generators import Singlet
import config as C


def model_path(name):
    return 'models/'+ name +'.model'


def get_convolutional_model():
    inputs_image_simple_convolutional = Input(shape=C.in_dim)

    x1 = BatchNormalization()(inputs_image_simple_convolutional) 
    x1 = Conv2D(32, (3, 3))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation='relu')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    #x1 = Dropout(0.1)(x1)

    x2 = Conv2D(48, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    #x2 = Dropout(0.1)(x2)

    x3 = Conv2D(64, (3, 3), padding='same')(x2)
    #x3 = BatchNormalization()(x3)
    x3 = Activation(activation='relu')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    #x3 = Dropout(0.1)(x3)
    x3 = Flatten()(x3)
    out = Dense(C.out_dim, activation='relu')(x3)
    return Model(inputs=inputs_image_simple_convolutional, outputs=out)

def get_inception_model():
    inp = Input(shape=C.in_dim)

    x1 = BatchNormalization()(inp) 
    x1 = Conv2D(3, (3, 3), padding='same')(x1)

    base_model = InceptionV3(weights='imagenet', include_top=False)(x1)

    tmp = GlobalAveragePooling2D()(base_model)

    return Model(inputs=inp, outputs=tmp)

def initialize_base_model():
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists(model_path(C.base_model)):
        print('Creating base network from scratch.')
        if C.base_model == 'simple_convolutional':
            return get_convolutional_model()
        elif C.base_model == 'inception':
            return get_inception_model()
    else:
        print('Loading model:'+model_path(C.base_model))
        return load_model(model_path(C.base_model))


def train_base_model(model, train_generator, val_generator ):
    inp = Input(shape=C.in_dim)
    x = model(inp)
    predictions = Dense(C.n_classes, activation='softmax', name="output")(x)
    trainable_model = Model(inputs=inp, outputs=predictions)
    trainable_model.compile(optimizer=SGD(lr=C.base_learn_rate, momentum=0.9),
                            loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])
    history = trainable_model.fit_generator(train_generator,
                                            validation_data=val_generator,
                                            validation_steps=C.base_validation_steps,
                                            epochs=C.base_epochs,
                                            steps_per_epoch=len(train_generator),
                                            workers=4,
                                            callbacks=[
                                                CSVLogger("history_trained_"+C.base_model),
                                                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1),
                                                EarlyStopping(monitor='val_loss', patience=20, verbose=1)
                                            ])
    return trainable_model


def main():
    model = initialize_base_model()
    print("Instantiating generators")
    train_generator = Singlet(
        batch_size=C.siamese_batch_size, directory=C.train_dir, steps_per_epoch=C.base_steps_per_epoch)
    val_generator = Singlet(
        batch_size=C.siamese_batch_size, directory=C.val_dir, steps_per_epoch=C.base_validation_steps)
    print("Training model")
    trainable_model = train_base_model(model, train_generator, val_generator)
    model.save(model_path(C.base_model))
    trainable_model.save(model_path("trained_"+C.base_model))

    

if __name__ == "__main__":
    main()
