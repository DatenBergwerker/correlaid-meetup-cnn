from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, InputLayer, Flatten


def simple_cnn_model(input_shape: tuple):
    """
    This model consists of a very basic CNN to illustrate the network architecture.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32,
                     kernel_size=(2, 2),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# def bit_more_advanced_cnn()
