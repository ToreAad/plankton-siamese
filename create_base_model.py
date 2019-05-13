import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import CSVLogger
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, \
    Concatenate, Lambda, Conv2D, MaxPooling2D, Dropout
from keras import backend as K
from keras.utils import to_categorical

from generators import Singlet
import config as C


def model_path(name):
    return 'models/'+ name +'.model'


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
    out = Dense(C.out_dim, activation='relu')(x3)
    return Model(inputs=inputs_image_simple_convolutional, outputs=out)


def initialize_base_model():
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists(model_path(C.base_model)):
        print('Creating base network from scratch.')
        return get_convolutional_model()
    else:
        print('Loading model:'+model_path(C.base_model))
        return load_model(model_path(C.base_model))


def plot_history(outputfolder, history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training data loss')
    plt.plot(history.history['val_loss'], label='Validation data loss')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], label='Training data accuracy')
    plt.plot(history.history['val_acc'], label='Validation data accuracy')
    plt.legend()
    plt.title("Accuracy")
    plt.suptitle("Model performance - {}".format(C.base_model))

    plt.savefig(os.path.join(outputfolder, 'model_history.png'))
    plt.show()


def print_summary(outputfolder, model, val_generator):
    score = model.evaluate_generator(
        val_generator, steps=C.base_validation_steps)
    summary = "Validation loss: {}\nValidation accuracy: {}".format(
        score[0], score[1])
    print(summary)
    with open(os.path.join(outputfolder, 'model_score.txt'), 'w+') as f:
        f.write(summary)


def plot_confusion_matrix(outputfolder, model, val_generator):
    img_vals = []
    stat_vals = []
    y_vals = []
    for i in range(C.base_validation_steps):
        x, y = val_generator[i]
        if len(x) == 2:
            img_vals.append(x[0])
            stat_vals.append(x[1])
        else:
            img_vals.append(x)
        y_vals.append(y)
    if stat_vals:
        X_validation = [np.concatenate(img_vals), np.concatenate(stat_vals)]
    else:
        X_validation = [np.concatenate(img_vals)]
    y_target = np.concatenate(y_vals)
    y_predicted = np.argmax(model.predict(X_validation), axis=1)
    cm = confusion_matrix(y_target, y_predicted)
    for i in range(len(cm)):
        cm[i, i] = 0

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, cmap="Greys")
    plt.xticks(range(max(y_target+1)))
    plt.yticks(range(max(y_target+1)))
    plt.title("Confusion matrix")
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.savefig(os.path.join(outputfolder, 'model_confusion_matrix.png'))
    plt.show()


def visualize_training(history, model, val_generator, iteration=""):
    outputfolder = C.base_model
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    plot_history(outputfolder, history)
    print_summary(outputfolder, model, val_generator)
    plot_confusion_matrix(outputfolder, model, val_generator)


def train_base_model(model, train_generator, val_generator ):
    inp = Input(shape=C.in_dim)
    x = model(inp)
    predictions = Dense(C.n_classes, activation='softmax', name="output")(x)
    trainable_model = Model(inputs=inp, outputs=predictions)
    trainable_model.compile(optimizer='rmsprop',
                            loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])
    history = trainable_model.fit_generator(train_generator,
                                            validation_data=val_generator,
                                            validation_steps=C.base_validation_steps,
                                            epochs=C.base_epochs,
                                            workers=4,
                                            callbacks=[
                                                CSVLogger(
                                                    C.logfile, append=True, separator='\t')
                                            ])
    return history, trainable_model


def main():
    model = initialize_base_model()
    print("Instantiating generators")
    train_generator = Singlet(
        batch_size=C.siamese_batch_size, directory=C.train_dir, steps_per_epoch=C.base_steps_per_epoch)
    val_generator = Singlet(
        batch_size=C.siamese_batch_size, directory=C.val_dir, steps_per_epoch=C.base_validation_steps)
    print("Training model")
    history, trainable_model = train_base_model(model, train_generator, val_generator)
    model.save(model_path(C.base_model))
    print("Visualizing training")
    visualize_training(history, trainable_model, val_generator)
    

if __name__ == "__main__":
    main()
