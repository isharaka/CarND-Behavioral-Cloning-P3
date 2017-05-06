import csv
import cv2
import numpy as np
import sklearn
import argparse

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from model import preprocess

datadir = 'data/small/'

lines = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()
    with open(datadir+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    angles = []
    for line in lines:
        name = datadir + './IMG/'+line[0].split('/')[-1]
        center_image = preprocess(cv2.imread(name))
        center_angle = float(line[3])
        images.append(center_image)
        angles.append(center_angle)

    # trim image to only see section with road
    X_train = np.array(images)
    y_train = np.array(angles)

    print(X_train.shape)
    print(X_train[[0]].shape)

    model = load_model(args.model)

    for i in range(X_train.shape[0]):
        predicted_angle = float(model.predict(X_train[[i]], batch_size=1))
        print(str(y_train[i])+" "+str(predicted_angle))





