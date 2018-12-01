from keras.datasets import cifar10
from keras.utils import to_categorical


from models.simple_cnn import simple_cnn_model
from models.cifar10_pretrained import cifar10_cnn

# dimensions = (N, 32 * 32 * 3)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 25


model = simple_cnn_model((32, 32, 3))

model.fit(x=x_train, y=y_train,
          epochs=15, batch_size=256)
model.evaluate(x=x_test, y=y_test)
model_path = os.path.join('models/saved', 'keras_simple_cnn_cifar10.h5')
model.save(model_path)

model = cifar10_cnn()
model.load_weights('models/saved/keras_cifar10_trained_model.h5')
model.evaluate(x=x_test, y=y_test)