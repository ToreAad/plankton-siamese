from keras.models import Model, Input
from keras.layers import Concatenate
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.optimizers import SGD

from create_base_model import initialize_base_model, model_path
from generators import triplet_generator

import config as C
import testing as T

def tripletize(bmodel):
    anc_in = Input(shape=C.in_dim)
    pos_in = Input(shape=C.in_dim)
    neg_in = Input(shape=C.in_dim)

    anc_out = bmodel(anc_in)
    pos_out = bmodel(pos_in)
    neg_out = bmodel(neg_in)

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


def train_siamese_model(model):
    model.compile(optimizer=SGD(lr=C.learn_rate, momentum=0.9),
              loss=std_triplet_loss())
    
    history = model.fit_generator(
        triplet_generator(C.batch_size, None, C.train_dir), 
        steps_per_epoch=1000, 
        epochs=C.iterations,
        callbacks=[
            CSVLogger(C.logfile, append=True, separator='\t')
        ],
        validation_data=triplet_generator(C.batch_size, None, C.val_dir), validation_steps=100)
    
    return history


def avg(x):
    return sum(x)/len(x)


def log(s):
    with open(C.logfile, 'a') as f:
        print(s, file=f)


# def summarize_progress():
#     vs = T.get_vectors(base_model, C.val_dir)
#     c = T.count_nearest_centroid(vs)
#     log('Summarizing '+str(i))
#     with open('summarize.'+str(i)+'.log', 'w') as sumfile:
#         T.summarize(vs, outfile=sumfile)
#     with open('clusters.'+str(i)+'.log', 'w') as cfile:
#         T.confusion_counts(c, outfile=cfile)
#     c_tmp = {}
#     r_tmp = {}
#     for v in vs:
#         c_tmp[v] = T.centroid(vs[v])
#         r_tmp[v] = T.radius(c_tmp[v], vs[v])
#     c_rad = [round(100*r_tmp[v])/100 for v in vs]
#     c_mv = [round(100*T.dist(c_tmp[v], cents[v]))/100 for v in vs]
#     log('Centroid radius: '+str(c_rad))
#     log('Centroid moved: '+str(c_mv))
#     cents = c_tmp

#     with open(C.logfile, 'a') as f:
#         T.accuracy_counts(c, outfile=f)
#     # todo: avg cluster radius, avg cluster distances
#     log('Avg centr rad: %.2f move: %.2f' % (avg(c_rad), avg(c_mv)))

def main():
    base_model = initialize_base_model()
    siamese_model = tripletize(base_model)

    vs = T.get_vectors(base_model, C.val_dir)
    cents = {}
    for v in vs:
        cents[v] = T.centroid(vs[v])

    learning_rate = C.learn_rate
    for i in range(C.last+1, C.last+11):
        log('Starting iteration '+str(i)+'/'+str(C.last+10)+' lr='+str(learning_rate))
        history = train_siamese_model(siamese_model)
        learning_rate = learning_rate * C.lr_decay
        base_model.save(model_path(C.base_model, i))
        # summarize_progress()

if __name__ == "__main__":
    main()