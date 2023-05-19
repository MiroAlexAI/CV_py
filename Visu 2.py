import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Создаем фигуру
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Создаем данные
x = np.arange(-10, 10, 0.1)
y = np.sin(x)

# Создаем график
graph, = plt.plot(x, y, lw=2, color='red')

# Создаем слайдер
ax_n = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_n = Slider(ax_n, 'n', -10, 10, valinit=0)

# Функция для обновления графика
def update(val):
    n = slider_n.val
    graph.set_ydata(np.sin(x * n))
    fig.canvas.draw_idle()

# Связываем слайдер с функцией
slider_n.on_changed(update)
plt.show()