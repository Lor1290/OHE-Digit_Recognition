from pickletools import optimize
import tensorflow as tf 
from tensorflow import keras
import numpy as np 

EPOCHS = 200
BATCH_SIZE = 128
N_HIDDEN = 128
CLASSES = 10
VERBOSE = 1

VAL_SPLIT = 0.2

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
RESHAPED = 28*28

x_train = x_train.reshape(60_000, RESHAPED)
x_test = x_test.reshape(10_000, RESHAPED)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

print("x_train sample:",x_train.shape[0])
print("x_test sample: ", x_test.shape[0])

y_train = tf.keras.utils.to_categorical(y_train, CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, CLASSES)

model = keras.models.Sequential()
model.add(keras.layers.Dense( CLASSES, 
                             input_shape=(RESHAPED,), 
                             name="dense_layer", 
                             activation="softmax"))
model.compile(optimizer="SGD", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

model.fit(x_train, y_train, 
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VAL_SPLIT)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


model.save("keras_model.h5")

tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_converter.optimization = [tf.lite.Optimize.DEFAULT] 
tf_lite_converter.target_spec.supported_types = [tf.float16] 
tensor_flow_model = tf_lite_converter.convert()

open("tf_lite_16.tflite", "wb").write(tensor_flow_model)
