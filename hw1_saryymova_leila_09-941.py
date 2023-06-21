import math
import numpy

points = []
edges = []
faces = []

with open('teapot.obj', 'rt') as f:
    for line in f:
        if line[0] == "v":
            line = line[1:]
            a = list(map(float, line.split()))
            points.append(a)
        else:
            line = line[1:]
            a = list(map(int, line.split()))
            faces.append(a)
faces.remove([])

for i in range(len(faces)):
    a = [faces[i][0], faces[i][1]]
    a.sort()
    if a in edges:
        pass
    else: edges.append(a)

    b = [faces[i][1], faces[i][2]]
    b.sort()
    if b in edges:
        pass
    else: edges.append(b)

    c = [faces[i][2], faces[i][0]]
    c.sort()
    if c in edges:
        pass
    else: edges.append(c)

sum = 0

for edge in edges:
    a = edge[0]
    b = edge[1]
    a1 = points[a-1]
    b1 = points[b-1]
    sum += math.sqrt((a1[0] - b1[0])**2 + (a1[1] - b1[1])**2 + (a1[2] - b1[2])**2)

print("Сумма всех векторов: ", sum)

data = numpy.zeros(len(points))

for i in range(len(faces)):
    a = faces[i][0]
    data[a-1] +=1
    a = faces[i][1]
    data[a-1] +=1
    a = faces[i][2]
    data[a-1] +=1

ind = numpy.argmax(data)
print("Наиболее часто встречающийся вектор:", points[ind])

