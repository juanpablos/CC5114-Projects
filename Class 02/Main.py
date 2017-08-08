import random
from perceptron.Perceptron import Perceptron
import matplotlib.pyplot as plt

def f(anX):
    return anX*4 + 1


canvas = 50
training = 100
points = 500


xx = list()
yy = list()

rx = list()
ry = list()
bx = list()
by = list()

per = Perceptron([1., 5.], -2.)

print(per.weights)

for j in range(0,training):
    x1 = random.uniform(-canvas, canvas)
    y1 = random.uniform(-canvas, canvas)

    if f(x1) < y1:
        expected = 1
    else:
        expected = 0

    res = per.train([x1, y1], expected)

print(per.weights)

for i in range(0,points):
    x = random.uniform(-canvas, canvas)
    y = random.uniform(-canvas, canvas)

    res = per.evaluate([x, y])

    if res:
        rx.append(x)
        ry.append(y)
    else:
        bx.append(x)
        by.append(y)

for j in range(int(-canvas/2), int(canvas/2)):
    xx.append(j)
    yy.append(f(j))


plt.plot(rx, ry, 'ro', bx, by, 'bo', xx, yy, 'g')
plt.show()
