import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam, adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import load_model
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import argparse
import ast
import random

np.random.seed(0)
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 150, 400, 1
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
batch_size = 50
nb_classes = 3

def load_data(args):

    data_df = pd.read_csv('log.csv', names=['image','input'])
    X1 = data_df[['image']].values
    Y1 = data_df['input'].values

    X = []
    Y = []
    p=0
    for add, out in zip(X1,Y1):
        img = cv2.imread(add[0], 0)
        k = ast.literal_eval(out)
        if k == [0,0,1]:
            X.append(img)
            Y.append(k)
            X.append(cv2.flip( img, 1 ))
            Y.append([1,0,0])
        elif k==[1,0,0]:
            X.append(img)
            Y.append(k)
            X.append(cv2.flip( img, 1 ))
            Y.append([0,0,1])

        elif k == [0,1,0]:
            if p % 3 == 0:
                X.append(img)
                Y.append(k)
            else:
                pass
        p += 1
    #print(Y)
    X = np.array(X, dtype=np.uint8)
    Y = np.array(Y, dtype=np.uint8)
    X = X.reshape(X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    X = X.astype('float32')
    X /= 255

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):

    '''model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(nb_classes))
    model.summary()
    '''
    activation_relu = 'relu'
    model = Sequential()

    #model.add(Lambda(lambda x: x / 127.5 - 1.0, )

    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2),input_shape=INPUT_SHAPE))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation("tanh"))

    model.add(Dense(100))
    model.add(Activation("tanh"))

    model.add(Dense(50))
    model.add(Activation("tanh"))

    model.add(Dense(10))
    model.add(Activation("tanh"))

    model.add(Dense(nb_classes,activation="softmax"))

    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """



    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')


    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.learning_rate))


    model.fit(X_train, y_train, epochs=args.nb_epoch,batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[checkpoint])


#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    #model = build_model(args)
    model = load_model("model1.h5")
    #train model on data, it saves as model.h5
    train_model(model, args, *data)


if __name__ == '__main__':
    main()