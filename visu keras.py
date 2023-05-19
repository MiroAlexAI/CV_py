import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import Model
import PIL

from keras.models import Sequential
from keras.layers import Dense


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def update(frame):
    line.set_ydata(np.sin(x + frame/10.0))
    return line,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
plt.show()

# generate binary numbers from 0 to 7
binary_list = [bin(x)[2:] for x in range(0, 8)]

# generate dataset
datasetB = []
datasetD = []
i = 0
for i in range(1000):
    padded_num = np.random.choice(binary_list)
    binary_number = padded_num.rjust(3, '0')

    decimal_number = int(binary_number, 2)
    datasetB.append(binary_number)
    datasetD.append(decimal_number)
X_train = datasetB
y_train = datasetD

datasetB = np.array(datasetB)

CLASS_COUNT = 8
# Преобразование ответов в формат one_hot_encoding
y_train = utils.to_categorical(y_train, CLASS_COUNT)



# print dataset
# print(binary_list)
# print(datasetB)
# print(datasetD)
#
from keras.callbacks import Callback
#
# class WeightsSaver(Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         self.model.save_weights('weights_epoch_{}.h5'.format(epoch))
#




# Создаем модель

model = Sequential()

# Добавляем слои
model.add(Dense(units=100, activation='relu', input_dim=1))
model.add(Dense(units=200))
model.add(Dense(units=4, activation='softmax'))

# Компилируем модель
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])



# Обучаем модель
model.fit(X_train, y_train, epochs=10, batch_size=5)

weight_tensors = [layer.weights[0] for layer in model.layers]

 # Теперь мы можем создать функцию, которая будет отображать изменения весов в процессе обучения:

def update_weights(frame_number):
            weights = [weight_tensor.numpy() for weight_tensor in weight_tensors]
            for ax, weight in zip(axes, weights):
                ax.cla()
                ax.imshow(weight)

        # Теперь, в зависимости от размера модели, мы можем создать график для разных весов и начать анимацию:

num_layers = len(weight_tensors)
fig, axes = plt.subplots(1, num_layers)

anim = animation.FuncAnimation(fig, update_weights, frames=100, interval=200)
plt.show()

