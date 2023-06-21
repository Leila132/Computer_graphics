import math
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'

def Drow(x, y, color, img):
    img[x, y] = color

def Bezier2(x0, y0, x1, y1, x2, y2, t):
    for i in t:
        P = [0, 0]
        P[0] = x0*(1-i)*(1-i) + 2*x1*(1-i)*i + x2*i*i 
        P[1] = y0*(1-i)*(1-i) + 2*y1*(1-i)*i + y2*i*i
        Drow(int(P[0]), int(P[1]), np.array([0,0,255], dtype=np.uint8), img)

n = 100 #количество кадров
frames = []
fig = plt.figure()

img = np.zeros((300, 300, 3), dtype = np.uint8)
t = np.linspace(0,1,1000)
PS = []
theta = np.linspace(0, 2 * np.pi, 1800)
r = 100
x = r * np.cos(theta) + 150
y = r * np.sin(theta) + 150

for i in range(20):
    PS.append([int(x[(90*i)%1800]), int(y[(90*i)%1800]),  int(x[(45 + 90*i)%1800]), int(y[(45 + 90*i)%1800]), int(x[(90 + 90*i)%1800]), int(y[(90 + 90*i)%1800])])

for i in range(len(PS)):
    Bezier2(PS[i][0], PS[i][1], PS[i][2], PS[i][3], PS[i][4], PS[i][5], t)
im = plt.imshow(img)
frames.append([im])

for f in range(n):
    img = np.zeros((300, 300, 3), dtype = np.uint8)
    if f in range(0, 25):
        for i in range(20):
            if i%2==0:
                PS[i][2] = (100+f*2) * np.cos(theta[45+90*i]) + 150
                PS[i][3] = (100+f*2) * np.sin(theta[45+90*i]) + 150
            else: 
                PS[i][2] = (100-f*2) * np.cos(theta[45+90*i]) + 150
                PS[i][3] = (100-f*2) * np.sin(theta[45+90*i]) + 150
    if f in range(25, 50):
        for i in range(20):
            if i%2==0:
                PS[i][2] = (199-f*2) * np.cos(theta[45+90*i]) + 150
                PS[i][3] = (199-f*2) * np.sin(theta[45+90*i]) + 150
            else: 
                PS[i][2] = f*2 * np.cos(theta[45+90*i]) + 150
                PS[i][3] = f*2 * np.sin(theta[45+90*i]) + 150
    if f in range(50, 75):
        for i in range(20):
            if i%2==0:
                PS[i][2] = (199-f*2) * np.cos(theta[45+90*i]) + 150
                PS[i][3] = (199-f*2) * np.sin(theta[45+90*i]) + 150
            else: 
                PS[i][2] = f*2 * np.cos(theta[45+90*i]) + 150
                PS[i][3] = f*2 * np.sin(theta[45+90*i]) + 150
    if f in range(75, 100):
        for i in range(20):
            if i%2==0:
                PS[i][2] = (f*2-100) * np.cos(theta[45+90*i]) + 150
                PS[i][3] = (f*2-100) * np.sin(theta[45+90*i]) + 150
            else: 
                PS[i][2] = (300-f*2) * np.cos(theta[45+90*i]) + 150
                PS[i][3] = (300-f*2) * np.sin(theta[45+90*i]) + 150
    for i in range(len(PS)):
        Bezier2(PS[i][0], PS[i][1], PS[i][2], PS[i][3], PS[i][4], PS[i][5], t)
    im = plt.imshow(img)
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
writer = PillowWriter(fps=24)
ani.save("line.gif", writer=writer)

plt.show()