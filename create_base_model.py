import os

from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, \
    Concatenate, Lambda, Conv2D, MaxPooling2D, Dropout
from keras import backend as K

from generators import singlet_generator
import config as C


def model_path(name, last):
    return 'models/'+str(name)+'_'+str(last)+'.model'


def get_convolutional_model():
    inputs_image_simple_convolutional = Input(shape=C.in_dim)

    x1 = Conv2D(32, (3, 3), activation='relu')(
        inputs_image_simple_convolutional)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Dropout(0.1)(x1)

    x2 = Conv2D(48, (3, 3), padding='same', activation='relu')(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.1)(x2)

    x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = Dropout(0.1)(x3)
    x3 = Flatten()(x3)
    x3 = Dense(512, activation='relu')(x3)
    bitvector = Dense(C.out_dim, activation='relu')(x3)
    return Model(inputs=inputs_image_simple_convolutional, outputs=bitvector)


def initialize_base_model():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(model_path(C.base_model, C.last)):
        print('Creating base network from scratch.')
        return get_convolutional_model()
    else:
        print('Loading model:'+model_path(C.base_model, C.last))
        return load_model(model_path(C.base_model, C.last))


def train_base_model(model):
    predictions = Dense(C.n_classes, activation='softmax',
                        name="output")(model)
    trainable_model = Model(inputs=model, outputs=predictions)
    trainable_model.compile(optimizer='rmsprop',
                            loss="categorical_crossentropy",
                            metrics=["accuracy"])

    train_generator = singlet_generator(
        batch_size=C.batch_size, directory=C.train_dir)
    val_generator = singlet_generator(
        batch_size=C.batch_size, directory=C.val_dir)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=C.steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=C.val_per_epoch,
                                  epochs=C.epochs,
                                  verbose=1)
    # callbacks=cbs,
    # verbose=1,
    # use_multiprocessing=True,
    # workers=5)
    return history


def main():
    model = initialize_base_model()
    history = train_base_model(model)
    model.save(C.base_model)
    return history


if __name__ == "__main__":
    main()
