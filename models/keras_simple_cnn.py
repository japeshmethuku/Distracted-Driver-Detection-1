# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:28:31 2019

@author: Hack5CHETeam14
"""
import numpy as np
np.random.seed(2016)

import matplotlib.pyplot as plt
import os
import glob
import cv2
import math
import pickle
import pandas as pd
import datetime
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import log_loss
from keras.models import model_from_json

dataset_path = "D:\\Aricent\\AIML\\Distracted-Driver-Detection\\state-farm-distracted-driver-detection"

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 96))
    return resized

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(dataset_path, 'imgs', 'train', 'c' + str(j), '*.jpg')
        print(path)
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)
    return X_train, y_train

def load_camera_img():
    print('Read images from camera')
    path = os.path.join(dataset_path, 'input.jpg')
    print(path)
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    #total = 0
    #thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        print(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        #total += 1
        img=cv2.imread(path,0)
        plt.imshow(img)
    return X_test, X_test_id


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout

def read_model():
    model = model_from_json(open(os.path.join(dataset_path, 'cache', 'architecture.json')).read())
    model.load_weights(os.path.join(dataset_path, 'cache', 'model_weights.h5'))
    return model

def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir(os.path.join(dataset_path,'cache')):
        os.mkdir(os.path.join(dataset_path,'cache'))
    open(os.path.join(dataset_path,'cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join(dataset_path,'cache', 'model_weights.h5'), overwrite=True)
    
def validate_holdout(model, holdout, target):
    predictions = model.predict(holdout, batch_size=128, verbose=1)
    score = log_loss(target, predictions)
    print('Score log_loss: ', score)
    # score = model.evaluate(holdout, target, show_accuracy=True, verbose=0)
    # print('Score holdout: ', score)
    # score = mlogloss(target, predictions)
    # print('Score : mlogloss', score)
    return score
    
def create_submission(predictions, test_id, loss):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(round(loss, 6)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

#import playsound
#def beep_function():
    #playsound.playsound('C:\\hack\\beep.mp3', True)
    
def classify_image(predications_new):
    for j in range(0,1):
      max_val = max(predictions_new[j])
      if max(predictions_new[j]) != (predictions_new[j][0]):
        print ("WARNING !!! Test Driver %s Distracted." %(j+1))
        if (max_val == (predictions_new[j][1])):
            print ("            Texting in phone (Right hand).")
            result = "Texting in phone (Right hand)"
        if (max_val == (predictions_new[j][2])):
            print ("            Speaking in phone (Right hand).")
            result = "Speaking in phone (Right hand)"
        if (max_val == (predictions_new[j][3])):
            print ("            Texting in phone (Left Hand).")
            result = "Texting in phone (Left hand)"
        if (max_val == (predictions_new[j][4])):
            print ("            Speaking in phone (Left Hand).")
            result = "Speaking in phone (Left hand)"
        if (max_val == (predictions_new[j][5])):
            print ("            Adjusting car items..")
            result = "Adjusting car items"
        if (max_val == (predictions_new[j][6])):
            print ("            Drinking..")
            result = "Drinking.."
        if (max_val == (predictions_new[j][7])):
            print ("            Turning.")
            result = "Turning.."
        if (max_val == (predictions_new[j][8])):
            print ("            Hand-Shakes.")
            result = "Hand-shakes"
        if (max_val == (predictions_new[j][9])):
            print ("            SPEAKING with co-passengers..")
            result = "SPEAKING with co-passengers"
      else:
        print("Driver %s NOT Distracted" %(j+1))
        result = "normal driving"
        return result

def display_result(img, result_string):
    #beep_function()
    #X_orig, y_orig = load_orig_camera_image()
    # Create a black image
    #img = np.zeros((512,512,3), np.uint8)

    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (150,150)
    fontScale              = 2
    fontColor              = (0,255,0)
    lineType               = 5

    cv2.putText(img, result_string, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType,
        cv2.LINE_AA)

    #Display the image
    cv2.imshow("img",img)
    #cv2.waitKey(0)


cache_path = os.path.join('cache', 'train.dat')
print (cache_path)

if not os.path.isfile(cache_path):
    print ("cache path doesnt exist previously")
    train_data, train_target = load_train()
    cache_data((train_data, train_target), cache_path)
else:
    print ("cache path exist previously")
    (train_data, train_target) = restore_data(cache_path)

img_rows, img_cols = 96, 128
nb_classes = 10
batch_size = 64

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
nb_epoch = 1

plt.imshow(train_data[0])

train_data = np.array(train_data,dtype=np.uint8)
train_target = np.array(train_target,dtype=np.uint8)

train_data = train_data.reshape(train_data.shape[0],img_rows,img_cols,1)
train_target = np_utils.to_categorical(train_target, nb_classes)
print (train_target[0])

train_data = train_data.astype('float32')
train_data /= 255
print('Train shape:', train_data.shape)
print(train_data.shape[0], 'train samples')

X_train, X_test, X_holdout, Y_train, Y_test, Y_holdout = split_validation_set_with_hold_out(train_data, train_target, 0.2)
print('Split train: ', len(X_train))
print('Split valid: ', len(X_test))
print('Split holdout: ', len(X_holdout))

#Building the Model
model_from_cache = 1

if model_from_cache == 1:
    model = read_model()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
else:
    model = Sequential()
    model.add(Conv2D(nb_filters, nb_conv, strides=(1,1), activation='relu',
                            padding='valid',
                            input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(nb_filters, nb_conv, activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

#compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#train the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
               verbose=1, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Score: ', score)
    score = model.evaluate(X_holdout, Y_holdout, verbose=0)
    print('Score holdout: ', score)
    
    validate_holdout(model, X_holdout, Y_holdout)
#save the model in cache
    save_model(model)

'''
LOAD IMAGE DYNAMIC
'''
test_data_new, test_id_new = load_camera_img()

test_data_new = np.array(test_data_new, dtype=np.uint8)
test_data_new = test_data_new.reshape(test_data_new.shape[0], img_rows, img_cols, 1)
# test_data = test_data.transpose((0, 3, 1, 2))
test_data_new = test_data_new.astype('float32')
test_data_new /= 255
print('Test shape:', test_data_new.shape)
print(test_data_new.shape[0], 'test samples')

#Predict Model
predictions_new = model.predict(test_data_new, batch_size=128, verbose=1)

display_string = classify_image(predictions_new)
