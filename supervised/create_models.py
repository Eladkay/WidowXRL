import pickle
import numpy as np
import cv2
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, RandomFlip, \
    RandomRotation, RandomZoom, Rescaling
import imutils
from tensorflow import keras

from rl_project.config import resize_factor


def create_both():
    # Load the data
    with open('labels.pickle', 'rb') as f:
        Y = pickle.load(f)  # format: y, x
        Y_xy = [(int(x / resize_factor), int(y / resize_factor)) for [y, x] in Y]
    X = [cv2.imread('dataset_binary/' + str(i) + '.png') / 255 for i in range(len(Y))]
    w, h = int(X[0].shape[1] / resize_factor), int(X[0].shape[0] / resize_factor)
    X = [imutils.resize(img, width=w) for img in X]

    # create models
    print(f'Creating model: xy_model_alt.h5')
    # split the data
    x_train, x_val, x_test = X[:int(len(X) * 0.7)], X[int(len(X) * 0.7):int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    y_train, y_val, y_test = Y[:int(len(Y) * 0.7)], Y[int(len(Y) * 0.7):int(len(Y) * 0.8)], Y[int(len(Y) * 0.8):]
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # create model
    data_augmentation = keras.Sequential(
        [
            RandomFlip("horizontal", input_shape=(h - 1, w, 3)),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ]
    )
    model = Sequential([
        data_augmentation,
        Rescaling(1. / 255),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(Y_xy))
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy,
                  metrics=['accuracy']
                  )

    # train model
    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val))

    # evaluation
    print('\nEvaluation:')
    model.evaluate(x_test, y_test)

    # prediction
    print('\nPrediction:')
    pred = model.predict(x_test)
    preds = [(np.argmax(pred[i]), y_test[i]) for i in range(len(pred))]
    print(preds)

    model.save("xy_model_alt.h5")


if __name__ == '__main__':
    create_both()
