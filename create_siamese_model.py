import os
import pickle

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Concatenate, Dense, BatchNormalization, Input
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

from create_base_model import initialize_base_model, freeze
from generators import Triplet, HierarchyTriplet

import config as C
import testing as T


def model_path(name, iteration=""):
    return 'models/' + name + '.model' if not iteration else 'models/' + name + '_'+iteration+'.model'


def initialize_bitvector_model():
    if not os.path.exists('models'):
        os.makedirs('models')

    path = model_path("bitvector_"+C.base_model)
    if True:
        print('Creating bitvector network from scratch.')
        model = initialize_base_model()
        m_in = Input(shape=C.in_dim)
        x = model(m_in)
        bitvector = Dense(C.out_dim, activation='sigmoid')(x)
        return Model(inputs=m_in, outputs=bitvector)
    else:
        print('Loading model:'+path)
        return load_model(path)


def tripletize(bitvector_model):
    anc_in = Input(shape=C.in_dim)
    pos_in = Input(shape=C.in_dim)
    neg_in = Input(shape=C.in_dim)

    anc_out = bitvector_model(anc_in)
    pos_out = bitvector_model(pos_in)
    neg_out = bitvector_model(neg_in)

    out_vector = Concatenate()([anc_out, pos_out, neg_out])
    return Model(inputs=[anc_in, pos_in, neg_in], outputs=out_vector)


def std_triplet_loss(alpha=5):
    """
    Basic triplet loss.
    Note, due to the K.maximum, this learns nothing when dneg>dpos+alpha
    """
    def myloss(y_true, y_pred):
        anchor = y_pred[:, 0:C.out_dim]
        pos = y_pred[:, C.out_dim:C.out_dim*2]
        neg = y_pred[:, C.out_dim*2:C.out_dim*3]
        pos_dist = K.sum(K.square(anchor-pos), axis=1)
        neg_dist = K.sum(K.square(anchor-neg), axis=1)
        basic_loss = pos_dist - neg_dist + alpha
        loss = K.maximum(basic_loss, 0.0)
        return loss

    return myloss


def hierarchy_triplet_loss(alpha=0.5):
    """

    """
    def hierarchyLoss(y_true, y_pred):
        anchor = y_pred[:, 0:C.out_dim]
        pos = y_pred[:, C.out_dim:C.out_dim*2]
        neg = y_pred[:, C.out_dim*2:C.out_dim*3]
        pos_dist = K.sqrt(K.sum(K.square(anchor-pos), axis=1))
        neg_dist = K.sqrt(K.sum(K.square(anchor-neg), axis=1))
        loss = pos_dist + alpha*K.abs(y_true - neg_dist)
        return loss
    return hierarchyLoss


def train_siamese_model(model, train_generator, val_generator, loss_function=std_triplet_loss):
    print("Starting to train")
    model.compile(optimizer=SGD(lr=C.learn_rate, momentum=0.9),
                  loss=loss_function())

    history = model.fit_generator(
        train_generator,
        epochs=C.siamese_epochs,
        callbacks=[
            CSVLogger("history_hierarchy_siamese_"+C.base_model),
            ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=3, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        ],
        validation_data=val_generator)

    return history


def main(generator=Triplet, loss_function=std_triplet_loss, out_name= "bitvector_"+C.base_model):
    bitvector_model = initialize_bitvector_model()
    siamese_model = tripletize(bitvector_model)
    train_generator = generator(
        batch_size=C.siamese_batch_size, directory=C.train_dir, steps_per_epoch=C.siamese_steps_per_epoch)
    val_generator = generator(
        batch_size=C.siamese_batch_size, directory=C.val_dir, steps_per_epoch=C.siamese_validation_steps)
    train_siamese_model(
        siamese_model, train_generator, val_generator, loss_function)
    freeze(bitvector_model).save(model_path(out_name))


if __name__ == "__main__":
    main(HierarchyTriplet, hierarchy_triplet_loss, "hierachy_bitvector_"+C.base_model )
