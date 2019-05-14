import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import time
import random as rn
from io import BytesIO

import redis

from tensorflow.keras.utils import Sequence


# from image_cache import toRedis, fromRedis
import config as C


def mk_triplets(directory):
    classes = os.listdir(directory)
    images = [os.listdir(os.path.join(directory, x)) for x in classes]

    while True:
        # pick random positive class
        pos_class = random.randint(0, len(classes)-1)
        # print('Anchor: ',pos_class,classes[pos_class])

        # pick random, different negative class
        neg_class = random.randint(0, len(classes)-2)
        if neg_class >= pos_class:
            neg_class = neg_class + 1
        # print('Negative: ',neg_class,classes[neg_class])

        # pick two random images from class
        anchor = os.path.join(
            directory, classes[pos_class], random.choice(images[pos_class]))
        pos = os.path.join(
            directory, classes[pos_class], random.choice(images[pos_class]))
        neg = os.path.join(
            directory, classes[neg_class], random.choice(images[neg_class]))

        # print('Selection:',anchor,pos,neg)
        yield (pos_class, neg_class, anchor, pos, neg)


# Scale to image size, paste on white background
def paste(img):
    i = np.ones(C.in_dim)
    # NB: Mono images lack the third dimension and will fail here:
    # (x,y,z) = img.shape
    (x, y) = img.shape
    start_x = int((299-x)/2)
    end_x = start_x + x
    start_y = int((299-y)/2)
    end_y = start_y + y
    i[start_x:end_x, start_y:end_y, 0] = img
    # i[start_x:end_x,start_y:end_y,1] = img
    # i[start_x:end_x,start_y:end_y,2] = img
    return i


def triplet_generator(batch_size, directory):
    trips = mk_triplets(directory)
    while True:
        ys = []
        ans = []
        pss = []
        ngs = []
        for _ in range(0, batch_size):
            pc, nc, anc, pos, neg = next(trips)
            ys.append((pc, nc))
            a_img = np.array(Image.open(anc))/256
            p_img = np.array(Image.open(pos))/256
            n_img = np.array(Image.open(neg))/256
            # Todo: paste it into the middle of a img_size'd canvas
            ans.append(paste(a_img))
            pss.append(paste(p_img))
            ngs.append(paste(n_img))
            # todo: augmentation

        a = np.asarray(ans)
        p = np.asarray(pss)
        n = np.asarray(ngs)
        y = np.asarray(ys)

        yield [a, p, n], y


def mk_singlets(directory):
    classes = os.listdir(directory)
    images = [os.listdir(os.path.join(directory, x)) for x in classes]

    while True:
        label = random.randint(0, len(classes)-1)
        image_path = os.path.join(
            directory, classes[label], random.choice(images[label]))
        yield (image_path, label)


def singlet_generator(batch_size, directory):
    trips = mk_singlets(directory)

    while True:
        images = []
        labels = []
        for _ in range(0, batch_size):
            image_path, label = next(trips)
            labels.append(label)
            image = np.array(Image.open(image_path))/256
            images.append(paste(image))
        X = np.asarray(images)
        y = np.asarray(labels)
        yield (X, y)


class Singlet(Sequence):
    def __init__(self, batch_size, directory, steps_per_epoch):
        self.batch_size = batch_size
        self.directory = directory
        self.steps_per_epoch = steps_per_epoch
        self.classes = os.listdir(directory)
        self.images = []
        self.r = redis.Redis(host='localhost', port=6379, db=0)
        # self.min_n = None
        for label in self.classes:
            image_names = os.listdir(os.path.join(directory, label))
            image_paths = [os.path.join(self.directory, label, name)
                           for name in image_names]
            # self.min_n = len(image_paths) if not self.min_n else min(
            #     len(image_paths), self.min_n)
            self.images.append(image_paths)
        self.seeded = False
        # self.X = {}
        # self.y = {}
        self.on_epoch_end()

    def paste(self, i, img):
        (x, y) = img.shape
        start_x = int((299-x)/2)
        end_x = start_x + x
        start_y = int((299-y)/2)
        end_y = start_y + y
        i[start_x:end_x, start_y:end_y, 0] = img

    def get_image(self, image_path):
        encoded = self.r.get(image_path)
        if not encoded:
            with open(image_path, "rb") as binary_file:
                encoded = binary_file.read()
                self.r.set(image_path, encoded)

        try:
            return np.array(Image.open(BytesIO(encoded)), dtype=np.float64)/256
        except:
            print("failed to store {}".format(image_path))
            return
            #self.r.delete(image_path)
            #return self.get_image(image_path)


    def on_epoch_end(self):
        if not self.seeded:
            rn.seed(int(time.time()*10000000) % 1000000007)
            np.random.seed(int(time.time()*10000000) % 1000000007)
            self.seeded = True


    def __len__(self):
        return self.steps_per_epoch


    def __getitem__(self, idx):
        images = np.ones((self.batch_size, *C.in_dim))
        labels = []
        for blank_img in images:
            label = random.randint(0, len(self.classes)-1)
            random_choice = random.randint(0, len(self.images[label])-1) #random.randint(0, self.min_n-1)
            image_path = self.images[label][random_choice]
            image = self.get_image(image_path)
            labels.append(label)
            self.paste(blank_img, image)
        return (np.asarray(images), np.asarray(labels))


class Triplet(Singlet):
    def __getitem__(self, idx):
        a_img = np.ones((self.batch_size, *C.in_dim))
        p_img = np.ones((self.batch_size, *C.in_dim))
        n_img = np.ones((self.batch_size, *C.in_dim))
        labels = []
        for j in range(self.batch_size):
            pos_class = random.randint(0, len(self.classes)-1)
            neg_class = random.randint(0, len(self.classes)-2)
            if neg_class >= pos_class:
                neg_class = neg_class + 1
            
            a_random_choice = random.randint(0, len(self.images[pos_class])-1)
            p_random_choice = random.randint(0, len(self.images[pos_class])-1)
            n_random_choice = random.randint(0, len(self.images[neg_class])-1)

            a_image_path = self.images[pos_class][a_random_choice]
            p_image_path = self.images[pos_class][p_random_choice]
            n_image_path = self.images[neg_class][n_random_choice]

            a_image = self.get_image(a_image_path)
            p_image = self.get_image(p_image_path)
            n_image = self.get_image(n_image_path)

            self.paste(a_img[j], a_image)
            self.paste(p_img[j], p_image)
            self.paste(n_img[j], n_image)
            labels.append((pos_class, neg_class))
            
        return( [a_img, p_img, n_img], np.asarray(labels))

# Testing:
if __name__ == "__main__":
    print("### Testing Singlet generator class ###")
    train_generator = Singlet(
        batch_size=C.base_batch_size, directory=C.train_dir, steps_per_epoch=9)
    for i in range(9):
        image, label = train_generator[i]
        for j in image:
            plt.imshow(j[:, :, 0])
            plt.show()

    print("### Testing Triplet generator class ###")
    train_generator = Triplet(
        batch_size=C.base_batch_size, directory=C.train_dir, steps_per_epoch=9)
    for i in range(9):
        [a_imgs, p_imgs, n_imgs], y = train_generator[i]
        for j in a_imgs:
            plt.imshow(j[:, :, 0])
            plt.show()
