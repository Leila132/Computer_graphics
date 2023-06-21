import math
import numpy as np
import matplotlib.pyplot as plt

points = []
faces = []
N = 700
with open('teapot.obj', 'rt') as f:
    for line in f:
        if line[0] == "v":
            line = line[1:]
            a = list(map(float, line.split()))
            a.remove(a[2])
            points.append(a)
        else:
            line = line[1:]
            a = list(map(int, line.split()))
            faces.append(a)
faces.remove([])
a = []
for i in range(len(points)):
    a.append(points[i][1])
a.sort()
print(a[0], a[-1])
print(points[1046], points[1054], points[952])
for i in range(len(points)):
    points[i][0] = int((points[i][0] + 3.5)*N/7)
    points[i][1] = int((points[i][1] + 3.5)*N/7)

img = np.zeros((N, N, 3), dtype = np.uint8)

def Drow(x, y, a):
    d = math.sqrt((x - (N/2))**2 + (y - (N/2))**2)
    if a: img[y, x] = [0, 255*(1-d/N), 0]
    else: img[x, y] = [0, 255, 0]
print("ffffffff", faces[209])
print(points[1046], points[1054], points[952])
def Brez(faces, points):
    for i in range(len(faces)):
        f1 = faces[i][0]
        f2 = faces[i][1]
        f3 = faces[i][2]

        x0 = points[f1-1][0]
        y0 = points[f1-1][1]
        x1 = points[f2-1][0]
        y1 = points[f2-1][1]
        a = False
        if x0>x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0>y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y1-y0 > x1-x0:
            x0, y0, x1, y1 = y0, x0, y1, x1
            a = True
        if abs(y1-y0) > abs(x1-x0):
            x0, y0, x1, y1 = y1, x1, y0, x0
            a =True
        ALG(x0, y0, x1, y1, a)
        a = False
        x0 = points[f2-1][0]
        y0 = points[f2-1][1]
        x1 = points[f3-1][0]
        y1 = points[f3-1][1]
        if x0>x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0>y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y1-y0 > x1-x0:
            x0, y0, x1, y1 = y0, x0, y1, x1
            a =True
        if abs(y1-y0) > abs(x1-x0):
            x0, y0, x1, y1 = y1, x1, y0, x0
            a =True
        ALG(x0, y0, x1, y1, a)
        a = False
        x0 = points[f1-1][0]
        y0 = points[f1-1][1]
        x1 = points[f3-1][0]
        y1 = points[f3-1][1]
        if x0>x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0>y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y1-y0 > x1-x0:
            x0, y0, x1, y1 = y0, x0, y1, x1
            a =True
        if abs(y1-y0) > abs(x1-x0):
            x0, y0, x1, y1 = y1, x1, y0, x0
            a =True
        ALG(x0, y0, x1, y1, a)

def ALG(x0, y0, x1, y1, a):
    delta_y = abs(y1 - y0)
    delta_x = abs(x1 - x0)
    error = 0.0
    y = y0
    for x in range(x0, x1+1):
        if a: Drow(y, x, a)
        else: Drow(x, y, a)
        error = error + delta_y
        if 2*error > delta_x:
            y += (1 if y1 > y0 else -1)
            error -= delta_x

Brez(faces, points)

plt.figure()
plt.imshow(img)
plt.show()