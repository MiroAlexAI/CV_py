import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
GPU_VISIBLE_DEVICES="0"
tf.compat.v1.enable_eager_execution()

#tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device('/GPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], ])


# Run on the GPU
c = tf.matmul(a, b)
print(c)

from tensorflow.keras import utils
binary_list = [bin(x)[2:] for x in range(0, 8)]

# generate dataset
datasetB = []
datasetD = []
i = 0
for i in range(600):
    padded_num = np.random.choice(binary_list)
    binary_number = float(padded_num)
    #binary_number = padded_num.rjust(3, '0')

    decimal_number = int(padded_num, 2)
    datasetB.append([binary_number])
    datasetD.append([decimal_number])
X_test = datasetB
y_test = datasetD

X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_test.shape)
print(y_test.shape)



# generate dataset
datasetB = []
datasetD = []
i = 0
for i in range(10000):
    padded_num = np.random.choice(binary_list)
    binary_number = float(padded_num)
    #binary_number = padded_num.rjust(3, '0')

    decimal_number = int(padded_num, 2)
    datasetB.append([binary_number])
    datasetD.append([decimal_number])
X_train = datasetB
y_train = datasetD

X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)
print(X_train)

CLASS_COUNT = 8
# Преобразование ответов в формат one_hot_encoding
y_train = utils.to_categorical(y_train, CLASS_COUNT)
y_test = utils.to_categorical(y_test, CLASS_COUNT)
print(y_train,y_test)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
# Создаем модель


model = Sequential()

# Добавляем слои
model.add(Dense(units=2, activation='elu', input_dim=1))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=8, activation='softmax'))

# Компилируем модель
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],)


#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

for layer in model.layers:
    tf.summary.histogram(layer.name + '/weights', layer.weights[0], step=0)
    tf.summary.histogram(layer.name + '/biases', layer.weights[1], step=0)

with tf.device('/CPU:0'):
# Обучаем модель
#  model.fit(X_train, y_train, epochs=200,validation_split=0.2, batch_size=22,callbacks=[tensorboard_callback])
    model.fit(X_train, y_train, epochs=30, validation_split=0.2, batch_size=24)
# fig, ax = plt.subplots()
# x = np.linspace(0, 2*np.pi, 100)
# line, = ax.plot(x, np.sin(x))
#
# def update(frame):
#     line.set_ydata(np.sin(x + frame/10.0))
#     return line,
#
# ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
# plt.show()