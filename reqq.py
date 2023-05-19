import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = sin(x)')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import time

for i in range(100):
    x = np.linspace(0, 4*np.pi, 150)
    y = np.sin(x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("y = sin(x)")
    plt.show()
    time.sleep(1)