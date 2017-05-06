import csv
import cv2
import numpy as np
import sklearn

datadir = 'data/'
folders = ['small/', 'small/']
lines = []

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

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = datadir +batch_sample[0].split('/')[-3]+ '/'+batch_sample[0].split('/')[-2]+ '/'+batch_sample[0].split('/')[-1]
                center_image = preprocess(cv2.imread(name))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

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
	model.add(Dropout(0.2))

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



if __name__ == '__main__':
	for f in range(len(folders)):
		readsamples(folders[f])
		print(len(lines))

	train_samples, validation_samples = train_test_split(lines, test_size=0.2)

	# compile and train the model using the generator function
	train_generator = generator(train_samples, batch_size=32)
	validation_generator = generator(validation_samples, batch_size=32)

	model = nvidia()
	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=50)

	model.save('model.h5')





