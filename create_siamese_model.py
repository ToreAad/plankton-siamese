from keras.models import Model, Input
from keras.layers import Concatenate, Dense, BatchNormalization
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.optimizers import SGD

from create_base_model import initialize_base_model
from generators import Triplet

import config as C
import testing as T


def model_path(name, iteration=""):
    return 'models/'+ name +'.model' if not iteration else 'models/'+ name +'_'+iteration+'.model'


def tripletize(bitvector_model):

    # m_in = Input(shape=C.in_dim)
    # x = model(m_in)
    # x = BatchNormalization()(x)
    # bitvector = Dense(C.out_dim, activation='relu')(x)
    # bitvector_model = Model(inputs=m_in, outputs=bitvector)
    
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
    # split the prediction vector
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


def avg(x):
    return sum(x)/len(x)


def log(s):
    with open(C.logfile, 'a') as f:
        print(s, file=f)


def train_siamese_model(model, train_generator, val_generator):
    model.compile(optimizer=SGD(lr=C.learn_rate, momentum=0.9),
                  loss=std_triplet_loss())

    history = model.fit_generator(
        train_generator,
        epochs=C.siamese_epochs,
        steps_per_epoch=len(train_generator),
        callbacks=[
            CSVLogger(C.logfile, append=True, separator='\t')
        ],
        validation_data=val_generator)

    return history


def summarizing_siamese_model(history, model, train_generator, val_generator, iteration="", cents={}):
    vs = T.get_vectors(model, C.val_dir)
    c = T.count_nearest_centroid(vs)
    log('Summarizing '+str(iteration))
    with open('summarize.'+iteration+'.log', 'w') as sumfile:
        T.summarize(vs, outfile=sumfile)
    with open('clusters.'+iteration+'.log', 'w') as cfile:
        T.confusion_counts(c, outfile=cfile)
    with open(C.logfile, 'a') as f:
        T.accuracy_counts(c, outfile=f)
    # todo: avg cluster radius, avg cluster distances

    c_tmp = {}
    r_tmp = {}
    for v in vs:
        c_tmp[v] = T.centroid(vs[v])
        r_tmp[v] = T.radius(c_tmp[v], vs[v])
    if cents:
        c_rad = [round(100*r_tmp[v])/100 for v in vs]
        c_mv = [round(100*T.dist(c_tmp[v], cents[v]))/100 for v in vs]
        log('Centroid radius: '+str(c_rad))
        log('Centroid moved: '+str(c_mv))
        log('Avg centr rad: %.2f move: %.2f' % (avg(c_rad), avg(c_mv)))
    cents = c_tmp

    return cents


def main():
    base_model = initialize_base_model()
    siamese_model = tripletize(base_model)
    train_generator = Triplet(
        batch_size=C.siamese_batch_size, directory=C.train_dir, steps_per_epoch=C.siamese_steps_per_epoch)
    val_generator = Triplet(
        batch_size=C.siamese_batch_size, directory=C.val_dir, steps_per_epoch=C.siamese_steps_per_epoch)
    history = train_siamese_model(
        siamese_model, train_generator, val_generator)
    base_model.save(model_path(C.base_model))
    siamese_model.save(model_path("siamese_"+C.base_model))
    # summarizing_siamese_model(history, base_model,
    #                           train_generator, val_generator)


if __name__ == "__main__":
    main()
