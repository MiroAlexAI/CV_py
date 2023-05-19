import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

n = int
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))


# Создаем слайдер
ax_n = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_n = Slider(ax_n, 'n', 0, 1000, valinit=20)

def animate(i):
    x = np.arange(0, 2*np.pi, 0.01)
    y = np.cos(x + i/10)
    line.set_data(x, y)
    return line,

def update(val):
    n = slider_n.val
    return int(n)

ani = animation.FuncAnimation(fig, animate, frames = update(n))




slider_n.on_changed(animate)
ani.save(filename='cos.gif')
plt.show()