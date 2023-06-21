import math
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'

N = 500 
n = 100 #количество кадров
def to_proj_coords1(x):
    r,c = x.shape
    x = np.concatenate([x, np.ones((1,c))], axis = 0)
    return x

def to_cart_coords1(x):
    x = x[:-1]/x[-1]
    return x
def shiftMatr(vec):
    mtr = np.array([[1, 0, vec[0]], [0, 1, vec[1]], [0, 0, 1]])
    return mtr

def rotMatr(ang):
    mtr = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return mtr
def bVec():
    b = np.array([0, 0])
    return b

def diagMatr(g):
    if g<0: mtr = np.array([[0, g], [g, 0]])
    else: mtr = np.array([[g, 0], [0, g]])
    return mtr

def detPos():
    mtr = np.array([[1.5, 1], [0.5, 1.2]])
    return mtr

def detNeg():
    mtr = np.array([[0.5, 1], [1.5, -1]])
    return mtr

def to_proj_coords(x):
    r,c = x.shape
    x = np.concatenate([x, np.ones((1,c))], axis = 0)
    return x

def to_cart_coords(x):
    x = x[:-1]/x[-1]
    return x


def ALG(x0, y0, x1, y1, color, img):
    a = False
    if x0>x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    
    if y1-y0 > x1-x0:
        x0, y0, x1, y1 = y0, x0, y1, x1
        a = True
    if abs(y1-y0) > abs(x1-x0):
        x0, y0, x1, y1 = y1, x1, y0, x0
        a = True
    delta_y = abs(y1 - y0)
    delta_x = abs(x1 - x0)
    error = 0.0
    y = y0
    for x in range(x0, x1+1):
        if a: Drow(y, x, color, img)
        else: Drow(x, y, color, img)
        error = error + delta_y
        if 2*error > delta_x:
            y += (1 if y1 > y0 else -1)
            error -= delta_x

def points(x0, y0, x1, y1):
    points = []
    a = False
    if x0>x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    
    if y1-y0 > x1-x0:
        x0, y0, x1, y1 = y0, x0, y1, x1
        a = True
    if abs(y1-y0) > abs(x1-x0):
        x0, y0, x1, y1 = y1, x1, y0, x0
        a = True
    delta_y = abs(y1 - y0)
    delta_x = abs(x1 - x0)
    error = 0.0
    y = y0
    for x in range(x0, x1+1):
        if a: points.append([int(y), int(x)])
        else: points.append([int(x), int(y)])
        error = error + delta_y
        if 2*error > delta_x:
            y += (1 if y1 > y0 else -1)
            error -= delta_x
    return points

def Drow(x, y, color, img):
    img[x, y] = color
theta = np.linspace(0, 2 * np.pi, 1000)
r = 100
x = r * np.cos(theta) + 150
y = r * np.sin(theta) + 150
circle =[]
for i in range(len(x)):
    circle.append([int(x[i]), int(y[i])])

def DrowSC(img): #Рисует круг и квадрат на их местах
    #Координаты квадраты
    x1_sq = 10
    y1_sq = 10
    x2_sq = 490
    y2_sq = 10
    x3_sq = 490
    y3_sq = 490
    x4_sq = 10
    y4_sq = 490
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = 100
    #Координаты круга
    x = r * np.cos(theta) + 150
    y = r * np.sin(theta) + 150
    for i in range(len(x)):
        img[int(x[i]), int(y[i])] = [250, 0, 0]

    ALG(x1_sq, y1_sq, x2_sq, y2_sq, np.array([0,255,0], dtype=np.uint8), img)
    ALG(x3_sq, y3_sq, x2_sq, y2_sq,np.array([0,255,0], dtype=np.uint8), img)
    ALG(x3_sq, y3_sq, x4_sq, y4_sq, np.array([0,255,0], dtype=np.uint8), img)
    ALG(x1_sq, y1_sq, x4_sq, y4_sq, np.array([0,255,0], dtype=np.uint8), img)

frames = []
fig = plt.figure()

r = 100
x1 = 500
y1 = 500
x1 = random.randint(250, 400)
y1 = random.randint(250, 400)
x2 = x1 + r
y2 = y1
x3 = int(x1 + r/2)
y3 = int(y1 + math.sqrt((3*r*r/4)))

R = 100*math.sqrt(3)/6
center = np.array([int(x1 + (r/2)), int(y1 + R)])

indi1 = True
indi2 = True

'''222'''
for i in range(n):
    img2 = np.zeros((N, N, 3), dtype = np.uint8)
    DrowSC(img2)
    pt1 = [x1, y1]
    pt2 = [x2, y2]
    pt3 = [x3, y3]
    R = 100*math.sqrt(3)/6          #радиус вписанной окружности
    center = np.array([int(x1 + (r/2)), int(y1 + R)])
    Drow(center[0], center[1], np.array([0,0,255], dtype=np.uint8), img2)
    x = np.array([pt1, pt2, pt3], dtype = np.float32).T
    x_proj = to_proj_coords(x)
    ang = i*2*np.pi/50
    T = shiftMatr(-center)
    R = rotMatr(ang)
    x_proj_new = np.linalg.inv(T) @ R @ T @ x_proj
    x_new = to_cart_coords(x_proj_new)
    pt1 = [x_new[0,0], x_new[1,0]]
    pt2 = [x_new[0,1], x_new[1,1]]
    pt3 = [x_new[0,2], x_new[1,2]]
    x = np.array([pt1, pt2, pt3], dtype = np.float32).T
    x_proj = to_proj_coords1(x)
    print(x_proj)

    if i%100 - 50 > 0: #увеличивается
        if indi1:
            x = np.array([pt1, pt2, pt3], dtype = np.float32).T
            x_proj = to_proj_coords1(x)
            a = diagMatr((i%100)/100)
            b = bVec() # change to non-zero
            m = np.zeros((3,3))
            m[:2,:2] = a
            m[:2,-1] = b
            m[-1,-1] = 1
            x_new_proj = m @ x_proj
            x_new = to_cart_coords(x_new_proj)
        else:
            x = np.array([pt1, pt2, pt3], dtype = np.float32).T
            x_proj = to_proj_coords1(x)
            a = diagMatr(-(i%100)/100)
            b = bVec() # change to non-zero
            m = np.zeros((3,3))
            m[:2,:2] = a
            m[:2,-1] = b
            m[-1,-1] = 1
            x_new_proj = m @ x_proj
            x_new = to_cart_coords(x_new_proj)            
    else: #уменьшается
        if indi2:
            x = np.array([pt1, pt2, pt3], dtype = np.float32).T
            x_proj = to_proj_coords1(x)
            a = diagMatr(1/(2**((i%51)/50)))
            b = bVec() # change to non-zero
            m = np.zeros((3,3))
            m[:2,:2] = a
            m[:2,-1] = b
            m[-1,-1] = 1
            x_new_proj = m @ x_proj
            x_new = to_cart_coords(x_new_proj)
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,1]), int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
        else:
            x = np.array([pt1, pt2, pt3], dtype = np.float32).T
            x_proj = to_proj_coords1(x)
            a = diagMatr(-1/(2**((i%51)/50)))
            b = bVec() # change to non-zero
            m = np.zeros((3,3))
            m[:2,:2] = a
            m[:2,-1] = b
            m[-1,-1] = 1
            x_new_proj = m @ x_proj
            x_new = to_cart_coords(x_new_proj)
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,1]), int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)            
    ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]), np.array([0,0,255], dtype=np.uint8), img2)
    ALG(int(x_new[0,1]), int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2]), np.array([0,0,255], dtype=np.uint8), img2)
    ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,2]), int(x_new[1,2]), np.array([0,0,255], dtype=np.uint8), img2)
    if i%99 == 0:
        x1, y1, x2, y2, x3, y3 = int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2])
    p1 = points(int(x_new[0,0]),int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]))
    p2 = points(int(x_new[0,1]),int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2]))
    p3 = points(int(x_new[0,0]),int(x_new[1,0]), int(x_new[0,2]), int(x_new[1,2]))
    
    for j in range(len(p1)):
        if p1[j] in circle:
            if i%100 - 50 < 0: #уменьшается
                indi2 = False
            else:
                indi1 = True
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,1]), int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
    for j in range(len(p2)):
        if p2[j] in circle:
            if i%100 - 50 < 0: #уменьшается
                indi2 = False
            else:
                indi1 = True
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,1]), int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
    for j in range(len(p3)):
        if p3[j] in circle:
            if i%100 - 50 < 0: #уменьшается
                indi2 = False
            else:
                indi1 = True
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,1]), int(x_new[1,1]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,1]), int(x_new[1,1]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)
            ALG(int(x_new[0,0]), int(x_new[1,0]), int(x_new[0,2]), int(x_new[1,2]), np.array([255,0,0], dtype=np.uint8), img2)



    im = plt.imshow(img2)
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=0)
writer = PillowWriter(fps=14)
ani.save("line.gif", writer=writer)

plt.show()