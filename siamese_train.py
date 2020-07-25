# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:15:40 2020

@author: asaga
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pickle as pkl
import cv2
import h5py
import time

from preprocess_data import dataloader

from tqdm import tqdm
from math import ceil

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.transform import rotate, AffineTransform, warp, rescale
from skimage.util import random_noise

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from statistics import mean

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

def pkl_data(filename):
    with open(filename,'rb') as f:
        X_t, y_t = pkl.load(f)
    return X_t, y_t


def affinetransform(image):
    transform = AffineTransform(translation=(-30,0))
    warp_image = warp(image,transform, mode="wrap")
    return warp_image

def anticlockwise_rotation(image):
    angle= random.randint(0,45)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,45)
    return rotate(image, -angle)


def transform(image):
    if random.random() > 0.5:
        image = affinetransform(image)
    if random.random() > 0.5:
        image = anticlockwise_rotation(image)
    if random.random() > 0.5:
        image = clockwise_rotation(image)

    return image



class data_gen:

    def __init__(self, batch_size = 32, isAug = True):
        self.batch_size = batch_size
        self.isAug = isAug

    def load_data_batch(self):

        training_file = 'training_file_183160.pkl'
        X,y = pkl_data(training_file)
        load_batch = 1024
        train_len = len(X)


        while(True):
            for i in range(int(train_len/load_batch)):
                start = i*load_batch
                end = (i+1)*load_batch if i != int(train_len/load_batch) else -1
                X_t = X[start:end]
                y_t = y[start:end]
                X_t, y_t = shuffle(X_t, y_t, random_state=0)

                for offset in range(0, load_batch, self.batch_size):
                    X_left, X_right, _y = X_t[offset:offset +self.batch_size,0],X_t[offset:offset + self.batch_size,1],y_t[offset:offset + self.batch_size]

                    #X_left, X_right, y = X_t[offset:offset +5,0],X_t[offset:offset + 5,1],y_t[offset:offset + 5]

                    X_left_batch = []
                    X_right_batch = []
                    y_batch = []


                    for i in range(len(X_left)):

                        if random.random() >1024:
                            X_i = np.expand_dims(transform(mpimg.imread(X_left[i])), axis = 2)
                            X_j = np.expand_dims(transform(mpimg.imread(X_right[i])), axis = 2)

                            X_left_batch.append(X_i)
                            X_right_batch.append(X_j)
                            y_batch.append(_y[i])
                        else:
                            X_i = np.expand_dims(mpimg.imread(X_left[i]), axis = 2)
                            X_j = np.expand_dims(mpimg.imread(X_right[i]), axis = 2)

                            X_left_batch.append(X_i)
                            X_right_batch.append(X_j)
                            y_batch.append(_y[i])

                    X_left_batch, X_right_batch, y_batch = np.asarray(X_left_batch), np.asarray(X_right_batch), np.asarray(y_batch)
                    X_left_batch, X_right_batch, y_batch  = shuffle(X_left_batch, X_right_batch, y_batch, random_state = 0)

                    #print("print_shape",X_left_batch.shape, X_right_batch.shape, y_batch.shape)
                    #print(X_left_batch[0], X_right_batch[1])

                    yield [X_left_batch, X_right_batch], y_batch
def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x-y), axis = 1, keepdims = True)
    result = K.maximum(sum_square, K.epsilon())
    return result

class siamese_network():
    def __init__(self, initial_learning_rate = 0.001, batch_size = 32):
        self.lr = initial_learning_rate
        self.batch_size = batch_size
        self.get_model()

    """
    def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x-y), axis = 1, keepdims = True)
    result = K.maximum(sum_square, K.epsilon())
    return result

    def l1_dist(vect):
         x, y = vect
         return K.abs(x-y)

    """

    def get_model(self):

        W_init_1 = RandomNormal(mean=0, stddev=0.01)
        b_init = RandomNormal(mean=0.5, stddev = 0.01)
        W_init_2 = RandomNormal(mean=0, stddev=0.2)

        input_shape = (105, 105, 1)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        convnet = Sequential()
        convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape, kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128,(7,7),activation='relu', kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128,(4,4),activation='relu', kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(256,(4,4),activation='relu', kernel_initializer=W_init_1, bias_initializer = b_init ,kernel_regularizer=l2(2e-4)))
        convnet.add(Flatten())
        convnet.add(Dense(4096,activation="sigmoid", kernel_initializer=W_init_2, bias_initializer = b_init ,kernel_regularizer=l2(1e-3)))
        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)

        merge_layer = Lambda(euclidean_dist)([encoded_l,encoded_r])
        prediction = Dense(1,activation='sigmoid')(merge_layer)
        self.model = Model(inputs=[left_input,right_input],outputs=prediction)

        optimizer = SGD(lr = 0.001, momentum = 0.5)

        """
        lr=3e-4, weight_decay=6e-5
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                    initial_learning_rate,
                                                                    decay_steps=10000,
                                                                    decay_rate=0.96,
                    lr_multipliers = {'Conv1': 0.01, 'Conv2':0.01, 'Conv3': 0.01, 'Conv4': 0.01, 'Dense1': 1}
        #opt = Adam(learning_rate = initial_learning_rate)
        opt=  Adam_dlr(lr = initial_learning_rate, lr_multipliers = lr_multipliers)taircase=True)
        """
        #lr_multipliers = {"Conv1": 1, "Conv2":1, "Conv3": 1, "Conv4": 1, "Dense1": 1}
        #opt =  Adam_dlr(learning_rate = 0.00006)
        #opt = SGD(lr = self.lr)
        self.model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])



    def test_pairs(self,  file_name ,n_way = 20):

        correct_pred = 0
        X,y = pkl_data(file_name)
        #print(X.shape, y.shape)

        j = 0
        for i in range(0,len(X),n_way):
            X_left, X_right,_y  = X[i: i+n_way,0],X[i: i+n_way,1], y[i : i+n_way]
            #X_left, X_right,y = sub_data_X[:,0], sub_data_X[:,1], sub_data_y
            X_left, X_right, _y = np.array(X_left), np.array(X_right), np.array(_y)

            correct_pred += self.test_one_shot(X_left, X_right, _y)


        acc =  correct_pred*100/(len(X)/n_way)
        return acc



    def test_one_shot(self, X_left,X_right, y):
        prob = self.model.predict([X_left,X_right])
        """
        print(prob)
        print(np.argmax(prob))
        print(np.argmax(y))
        return
        """
        if np.argmax(prob) == np.argmax(y):
            return 1
        else:
            return 0

    def test_validation_acc(self,wA_file, uA_file, n_way=20):
        wA_acc = self.test_pairs(wA_file,n_way)
        uA_acc = self.test_pairs(uA_file, n_way)
        return (wA_acc, uA_acc)

    def continue_training(self):

        with open('best_model/model_details.pkl','rb') as f:
            model_details = pkl.load(f)

        with open(self.val_acc_filename, "rb") as f:
            self.v_acc,self.train_metrics  = pkl.load(f)

        self.best_acc = model_details['acc']
        self.start = model_details['iter']+1
        K.set_value(self.model.optimizer.learning_rate, model_details['model_lr'])
        K.set_value(self.model.optimizer.momentum, model_details['model_mm'])
        best_model = 'best_model/best_model.h5'
        self.model.load_weights(best_model)
        print('\n\n----------------------------------------------------Loading saved Model----------------------------------------------------\n\n')


    def train_on_data(self, load_prev_model = False ,best_acc = 0):

        model_json = self.model.to_json()
        wA_file ='wA_val_10_split_images.pkl'
        uA_file ='uA_val_10_split_images.pkl'


        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        self.val_acc_filename = 'val_acc'

        self.v_acc = []
        self.train_metrics = []
        self.best_acc = best_acc
        self.model_details = {}
        self.model_details['acc'] = 0
        self.model_details['iter'] = 0
        self.model_details['model_lr'] = 0.0
        self.model_details['model_mm'] = 0.0
        linear_inc = 0.01
        self.start = 1
        self.k = 0


        if load_prev_model:
            self.continue_training()

        data_generator = data_gen(self.batch_size, isAug = True)
        train_generator = data_generator.load_data_batch()

        train_loss, train_acc = [],[]
        for i in range(self.start,1000000):

            """
            if self.k==50:
                K.set_value(model.model.optimizer.learning_rate, K.get_value(model.model.optimizer.learning_rate) * 0.9)
                self.k = 0
            """

            start_time = time.time()
            X_batch, y_batch = next(train_generator)
            #print(X_batch[0].shape,X_batch[1].shape, y_batch.shape)
            #print(type(X_batch), type(y_batch))
            #return

            loss = self.model.train_on_batch(X_batch, y_batch)
            train_loss.append(loss[0])
            train_acc.append(loss[1])

            if i % 500 == 0:
                train_loss = mean(train_loss)
                train_acc = mean(train_acc)
                self.train_metrics.append([train_loss,train_acc])

                #loss_data.append(loss)

                val_acc  = self.test_validation_acc(wA_file, uA_file, n_way=20)
                #val_acc = [wA_acc, uA_acc]
                self.v_acc.append(val_acc)
                if val_acc[0] > self.best_acc:
                    print('\n***Saving model***\n')
                    #self.model.save_weights("model_{}_val_acc_{}.h5".format(i,val_acc[0]))
                    self.model.save_weights("best_model/best_model.h5".format(i,val_acc[0]))
                    self.model_details['acc'] = val_acc[0]
                    self.model_details['iter'] = i
                    self.model_details['model_lr'] = K.get_value(self.model.optimizer.learning_rate)
                    self.model_details['model_mm'] = K.get_value(self.model.optimizer.momentum)
                    #siamese_net.save(model_path)
                    self.best_acc = val_acc[0]
                    with open(self.val_acc_filename, "wb") as f:
                        pkl.dump((self.v_acc,self.train_metrics), f)
                    with open('best_model/model_details.pkl', "wb") as f:
                        pkl.dump(self.model_details, f)

                end_time = time.time()
                print('Iteration :{}  lr :{:.8f} momentum :{:.6f} avg_loss: {:.4f} avg_acc: {:.4f} wA_acc :{:.2f} %  u_Acc: {:.2f} % time_taken {:.2f} s'.format(i,K.get_value(self.model.optimizer.learning_rate),K.get_value(self.model.optimizer.momentum),train_loss, train_acc,val_acc[0], val_acc[1], end_time-start_time))

                #
                train_loss, train_acc = [],[]

            if i % 5000 == 0:
                K.set_value(self.model.optimizer.learning_rate, K.get_value(self.model.optimizer.learning_rate) * 0.99)
                K.set_value(self.model.optimizer.momentum, min(0.9,K.get_value(self.model.optimizer.momentum) + linear_inc))



if __name__ == "__main__":

    model = siamese_network(batch_size = 32)

    #182000
    #216000
    #model.train_on_data(load_prev_model = True)

    model.train_on_data(load_prev_model = False)
