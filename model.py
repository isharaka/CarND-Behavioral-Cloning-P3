import csv
import cv2
import numpy as np
import sklearn

datadir = 'data/'
folders = ['original/', 'recovery/', 'lap3/']
#folders = ['small/', 'small/']
lines = []

AUGMENT = False
LEARNRATE = 0.0005

def readsamples(folder):
    with open(datadir+folder+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

def preprocess(image):
    image = image[50:140, 0:320]
    image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
    return image

from sklearn.utils import shuffle

def generator(samples, batch_size=32, multicam=False, augment=False):
    num_samples = len(samples)
    delta_angles = [0.0, 0.1, -0.1]
    n_cameras = 3 if (multicam) else 1
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for cam in range(n_cameras):
                    name = datadir +batch_sample[cam].split('/')[-3]+ '/'+batch_sample[cam].split('/')[-2]+ '/'+batch_sample[cam].split('/')[-1]
                    image = preprocess(cv2.imread(name))
                    angle = float(batch_sample[3]) + delta_angles[cam]
                    images.append(image)
                    angles.append(angle)

                    if (augment == True):
                        rand = np.random.randint(0,99)

                        if(rand < 40):
                            # Flip the image and steering angle
                            image = cv2.flip(image,1)
                            angle = -1.0 * angle
                        elif(rand < 70):
                            # Remove left half of the image
                            image[0:66, 0:100] = 0
                        else:
                            # Remove right half of the image
                            image[0:66, 100:200] = 0

                        images.append(image)
                        angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU

def lenet():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode="valid", activation='linear'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode="valid", activation='linear'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode="valid", activation='linear'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", activation='linear'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", activation='linear'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(1))

    return model


from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pickle




if __name__ == '__main__':
    for f in range(len(folders)):
        readsamples(folders[f])
        print(len(lines))

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32, multicam=False, augment=AUGMENT)
    validation_generator = generator(validation_samples, batch_size=32, multicam=False)

    model = nvidia()
    adam_ptimizer = Adam(lr=LEARNRATE, decay=0.0000)

    model.compile(loss='mse', optimizer=adam_ptimizer)

    callback_check_point = ModelCheckpoint('./cp/model-epoch-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
    callback_early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

    training_samples = ( len(train_samples) * 2 ) if (AUGMENT) else len(train_samples)
    history_object = model.fit_generator(train_generator, samples_per_epoch=training_samples, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=100, callbacks=[callback_early_stop])

    model.save('model.h5')

    pickle.dump(history_object.history, open("history.p", "wb"))
    history = pickle.load(open("history.p", "rb"))

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    #fig = plt.figure()
    #plt.plot(history['loss'])
    #plt.plot(history['val_loss'])
    #plt.title('model mean squared error loss')
    #plt.ylabel('mean squared error loss')
    #plt.xlabel('epoch')
    #plt.legend(['training set', 'validation set'], loc='upper right')

    #plt.show()

    #fig.savefig('history.png')





