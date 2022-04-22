import pickle
import numpy as np
import cv2
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout
import imutils
from create_dataset import RESIZE_FACTOR


def create_model(X, Y, *, last_layer_size, epochs=1, model_name='model.h5'):
    print(f'Creating model: {model_name}')
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
    model = Sequential([
        Flatten(input_shape=X[0].shape),
        Dense(8192, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(last_layer_size, activation='softmax', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    # train model
    model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_val, y_val))

    # evaluation
    print('\nEvaluation:')
    model.evaluate(x_test, y_test)

    # prediction
    print('\nPrediction:')
    pred = model.predict(x_test)
    preds = [(np.argmax(pred[i]), y_test[i]) for i in range(len(pred))]
    print(preds)

    model.save(model_name)


def create_both():
    # Load the data
    with open('labels.pickle', 'rb') as f:
        Y = pickle.load(f)  # format: y, x
        Y_x = [int(x / RESIZE_FACTOR) for [_, x] in Y]
        Y_y = [int(y / RESIZE_FACTOR) for [y, _] in Y]
    X = [cv2.imread('dataset_binary/' + str(i) + '.png') / 255 for i in range(len(Y))]
    w, h = int(X[0].shape[1] / RESIZE_FACTOR), int(X[0].shape[0] / RESIZE_FACTOR)
    X = [imutils.resize(img, width=w) for img in X]

    # create models
    create_model(X, Y_x, last_layer_size=w, epochs=10, model_name='x_model.h5')
    create_model(X, Y_y, last_layer_size=h, epochs=10, model_name='y_model.h5')


if __name__ == '__main__':
    create_both()
