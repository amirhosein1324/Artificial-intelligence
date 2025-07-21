import random
import matplotlib.pyplot as plt
import numpy as np

x1 = np.random.randint(10 , 30 , 200)
x2 = np.random.randint(10 , 30 , 200)

plot = plt.scatter(x1,x2 , color = "black")

x1[x1 < 20] -= 5
x1[x1 >= 20] += 5
x2[x2 < 20] -= 5
x2[x2 >= 20] += 5

plot = plt.scatter(x1 , x2 , color = "black")
plt.title("Clustring")

Kx = np.random.randint(0 , 40 , 4)
Ky = np.random.randint(0 , 40 , 4)

plot = plt.scatter(x1 , x2 , color = "black")
plt.scatter(Kx , Ky , color = "red")

Dx = np.array([(((Kx[i] - x1) ** 2) + ((Ky[i] - x2) ** 2)) ** 0.5 for i in range(len(Kx))])
Dx
Dy = np.array([(())])

k1 = Dx[0] <= np.mean(Dx[0])
k2 = Dx[1] <= np.mean(Dx[1])
k3 = Dx[2] <= np.mean(Dx[2])
k4 = Dx[3] <= np.mean(Dx[3])

plt.scatter(x1[k1] , x2[k1] , color = "black")
plt.scatter(x1[k2] , x2[k2] , color = "green")
plt.scatter(x1[k3] , x2[k3] , color = "red")
plt.scatter(x1[k4] , x2[k4] , color = "blue")
plt.scatter(Kx , Ky , color = "purple")

x_distance = np.array([abs(Kx[x] - x1) for x in range(len(Kx))])
y_distance = np.array([abs(Ky[y] - x2) for y in range(len(Ky))])

distances = (x_distance ** 2 + y_distance ** 2) ** 0.5
distances = distances.T

K = np.full(shape = (200 , 4) , fill_value= False , dtype = bool)
for row in range(len(distances)):
  column = np.where(distances[row] == np.min(distances[row]))
  K[row , column] = True
K = K.T

plot = plt.scatter(x1[K[0]] , x2[K[0]] , color = "black")
plot = plt.scatter(x1[K[1]] , x2[K[1]] , color = "green")
plot = plt.scatter(x1[K[2]] , x2[K[2]] , color = "blue")
plot = plt.scatter(x1[K[3]] , x2[K[3]] , color = "red")
plot = plt.scatter(Kx , Ky , s = 200 , marker = "X" , color = "purple")
plt.title("K-Means Clustering Result")

Kx[0] = np.mean(x1[K[0]])
Ky[0] = np.mean(x2[K[0]])
Kx[1] = np.mean(x1[K[1]])
Ky[1] = np.mean(x2[K[1]])
Kx[2] = np.mean(x1[K[2]])
Ky[2] = np.mean(x2[K[2]])
Kx[3] = np.mean(x1[K[3]])
Ky[3] = np.mean(x2[K[3]])

plot = plt.scatter(x1[K[0]] , x2[K[0]] , color = "black")
plot = plt.scatter(x1[K[1]] , x2[K[1]] , color = "green")
plot = plt.scatter(x1[K[2]] , x2[K[2]] , color = "blue")
plot = plt.scatter(x1[K[3]] , x2[K[3]] , color = "red")
plot = plt.scatter(Kx , Ky , s = 200 , marker = "X" , color = "purple")
plt.title("K-Means Clustering Result")

