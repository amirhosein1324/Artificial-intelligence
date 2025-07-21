import numpy as np
import matplotlib.pyplot as plt

iris_array = np.loadtxt(fname="/content/IRIS (1).csv", skiprows=1, usecols=[0, 1, 2, 3], delimiter=",")

sepal_length = iris_array[:, 0]
sepal_width = iris_array[:, 1]
petal_length = iris_array[:, 2]
petal_width = iris_array[:, 3]

plot = plt.scatter(sepal_length, sepal_width)

sepal_kx = np.random.rand(3) * (max(sepal_length) - min(sepal_length)) + min(sepal_length)
sepal_ky = np.random.rand(3) * (max(sepal_width) - min(sepal_width)) + min(sepal_width)

plot = plt.scatter(sepal_length, sepal_width)
plot = plt.scatter(sepal_kx, sepal_ky, marker="x", s=200, color="red")

sepal_x_distances = np.array([abs(sepal_kx[x] - sepal_length) for x in range(len(sepal_kx))])
sepal_y_distances = np.array([abs(sepal_ky[y] - sepal_width) for y in range(len(sepal_ky))])
sepal_distances = np.transpose((sepal_x_distances ** 2 + sepal_y_distances ** 2) ** 0.5)

sepal_k = np.full(shape=(len(sepal_distances), len(sepal_kx)), fill_value=False, dtype=bool)
for row in range(len(sepal_distances)):
    col = np.where(sepal_distances[row] == np.min(sepal_distances[row]))
    sepal_k[row, col] = True
sepal_k = np.transpose(sepal_k)

plot = plt.scatter(sepal_length[sepal_k[0]], sepal_width[sepal_k[0]])
plot = plt.scatter(sepal_length[sepal_k[1]], sepal_width[sepal_k[1]])
plot = plt.scatter(sepal_length[sepal_k[2]], sepal_width[sepal_k[2]])
plot = plt.scatter(sepal_kx, sepal_ky, marker="x", color="red", s=200)

sepal_kx = np.array([np.mean(sepal_length[sepal_k[x]]) for x in range(len(sepal_k))])
sepal_ky = np.array([np.mean(sepal_width[sepal_k[y]]) for y in range(len(sepal_k))])

plot = plt.scatter(sepal_length[sepal_k[0]], sepal_width[sepal_k[0]])
plot = plt.scatter(sepal_length[sepal_k[1]], sepal_width[sepal_k[1]])
plot = plt.scatter(sepal_length[sepal_k[2]], sepal_width[sepal_k[2]])
plot = plt.scatter(sepal_kx, sepal_ky, marker="x", color="red", s=200)

def fix_sepal_centroids(sepal_length, sepal_width, sepal_kx, sepal_ky):
    sepal_x_distances = np.array([abs(sepal_kx[x] - sepal_length) for x in range(len(sepal_kx))])
    sepal_y_distances = np.array([abs(sepal_ky[y] - sepal_width) for y in range(len(sepal_ky))])
    sepal_distances = np.transpose((sepal_x_distances ** 2 + sepal_y_distances ** 2) ** 0.5)

    sepal_k = np.full(shape=(len(sepal_distances), len(sepal_kx)), fill_value=False, dtype=bool)
    for row in range(len(sepal_distances)):
        col = np.where(sepal_distances[row] == np.min(sepal_distances[row]))
        sepal_k[row, col] = True
    sepal_k = np.transpose(sepal_k)

    new_sepal_kx = np.array([np.mean(sepal_length[sepal_k[x]]) for x in range(len(sepal_k))])
    new_sepal_ky = np.array([np.mean(sepal_width[sepal_k[y]]) for y in range(len(sepal_k))])

    if all(new_sepal_kx == sepal_kx) and all(new_sepal_ky == sepal_ky):
        plot = plt.scatter(sepal_length[sepal_k[0]], sepal_width[sepal_k[0]] , color = "yellow")
        plot = plt.scatter(sepal_length[sepal_k[1]], sepal_width[sepal_k[1]] , color = "blue")
        plot = plt.scatter(sepal_length[sepal_k[2]], sepal_width[sepal_k[2]] , color = "black")
        plot = plt.scatter(sepal_kx, sepal_ky, marker="x", s=200 , color = "purple")
        return sepal_length, sepal_width, new_sepal_kx, new_sepal_ky

    else:
      return fix_sepal_centroids(sepal_length, sepal_width, new_sepal_kx, new_sepal_ky)

sepal_length, sepal_width, sepal_kx, sepal_ky = fix_sepal_centroids(sepal_length, sepal_width, sepal_kx, sepal_ky)

plot = plt.scatter(petal_length, petal_width)

petal_kx = np.random.rand(3) * (max(petal_length) - min(petal_length)) + min(petal_length)
petal_ky = np.random.rand(3) * (max(petal_width) - min(petal_width)) + min(petal_width)

plot = plt.scatter(petal_length, petal_width)
plot = plt.scatter(petal_kx, petal_ky, marker="x", color="red", s=200)

petal_x_distances = np.array([abs(petal_kx[x] - petal_length) for x in range(len(petal_kx))])
petal_y_distances = np.array([abs(petal_ky[y] - petal_width) for y in range(len(petal_ky))])
petal_distances = np.transpose((petal_x_distances ** 2 + petal_y_distances) ** 0.5)

petal_k = np.full(shape=(len(petal_distances), len(petal_kx)), fill_value=False, dtype=bool)
for row in range(len(petal_distances)):
    col = np.where(petal_distances[row] == np.min(petal_distances[row]))
    petal_k[row, col] = True
petal_k = np.transpose(petal_k)

plot = plt.scatter(petal_length[petal_k[0]], petal_width[petal_k[0]])
plot = plt.scatter(petal_length[petal_k[1]], petal_width[petal_k[1]])
plot = plt.scatter(petal_length[petal_k[2]], petal_width[petal_k[2]])
plot = plt.scatter(petal_kx, petal_ky, marker="x", color="red", s=200)

petal_kx = np.array([np.mean(petal_length[petal_k[x]]) for x in range(len(petal_k))])
petal_ky = np.array([np.mean(petal_width[petal_k[y]]) for y in range(len(petal_k))])

plot = plt.scatter(petal_length[petal_k[0]], petal_width[petal_k[0]])
plot = plt.scatter(petal_length[petal_k[1]], petal_width[petal_k[1]])
plot = plt.scatter(petal_length[petal_k[2]], petal_width[petal_k[2]])
plot = plt.scatter(petal_kx, petal_ky, marker="x", color="red", s=200)

def fix_petal_centroids(petal_length, petal_width, petal_kx, petal_ky):
    petal_x_distances = np.array([abs(petal_kx[x] - petal_length) for x in range(len(petal_kx))])
    petal_y_distances = np.array([abs(petal_ky[y] - petal_width) for y in range(len(petal_ky))])
    petal_distances = np.transpose((petal_x_distances ** 2 + petal_y_distances) ** 0.5)

    petal_k = np.full(shape=(len(petal_distances), len(petal_kx)), fill_value=False, dtype=bool)
    for row in range(len(petal_distances)):
        col = np.where(petal_distances[row] == np.min(petal_distances[row]))
        petal_k[row, col] = True
    petal_k = np.transpose(petal_k)

    new_petal_kx = np.array([np.mean(petal_length[petal_k[x]]) for x in range(len(petal_k))])
    new_petal_ky = np.array([np.mean(petal_width[petal_k[y]]) for y in range(len(petal_k))])

    if all(new_petal_kx == petal_kx) and all(new_petal_ky == petal_ky):
        plot = plt.scatter(petal_length[petal_k[0]], petal_width[petal_k[0]] , color = "yellow")
        plot = plt.scatter(petal_length[petal_k[1]], petal_width[petal_k[1]] , color = "blue")
        plot = plt.scatter(petal_length[petal_k[2]], petal_width[petal_k[2]] , color = "black")
        plot = plt.scatter(petal_kx, petal_ky, marker="x", color="purple", s=200)
        return petal_length, petal_width, new_petal_kx, new_petal_ky

    else:
        return fix_petal_centroids(petal_length, petal_width, new_petal_kx, new_petal_ky)

petal_length, petal_width, petal_kx, petal_ky = fix_petal_centroids(petal_length, petal_width, petal_kx, petal_ky)

#1 - random centroids
kx = np.random.rand(3) * (max(sepal_length) - min(sepal_length)) + min(sepal_length)
ky = np.random.rand(3) * (max(sepal_width) - min(sepal_width)) + min(sepal_width)
kz = np.random.rand(3) * (max(petal_length) - min(petal_length)) + min(petal_length)

def calculate_distances(kx, ky, kz):
    x_distances = np.array([abs(kx[x] - sepal_length) for x in range(len(kx))])
    y_distances = np.array([abs(ky[y] - sepal_width) for y in range(len(ky))])
    z_distances = np.array([abs(kz[z] - petal_length) for z in range(len(kz))])
    distances = np.transpose(np.linalg.norm([x_distances, y_distances, z_distances] , axis=0))
    return distances

def is_finished(distances, new_distances):
    # final_distances = distances - new_distances <= 1e-3
    return True if np.all(new_distances - distances <= 1e-3) else False

while True:
    #2 - determine which node belong to which centroid
    distances = calculate_distances(kx, ky, kz)
    k = np.full(shape=distances.shape, fill_value=False, dtype=bool)
    for row in range(len(distances)):
        col = np.where(distances[row] == np.min(distances[row]))
        k[row, col] = True
    k = np.transpose(k)

    #3 - change centroids location to mean of related nodes
    new_kx = np.array([np.mean(sepal_length[k[x]]) for x in range(len(k))])
    new_ky = np.array([np.mean(sepal_width[k[y]]) for y in range(len(k))])
    new_kz = np.array([np.mean(petal_length[k[z]]) for z in range(len(k))])
    new_distances = calculate_distances(new_kx, new_ky, new_kz)

    if is_finished(distances, new_distances):
        fig = plt.figure(figsize=(9 , 9))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sepal_length[k[0]], sepal_width[k[0]], petal_length[k[0]], color='yellow')
        ax.scatter(sepal_length[k[1]], sepal_width[k[1]], petal_length[k[1]], color='blue')
        ax.scatter(sepal_length[k[2]], sepal_width[k[2]], petal_length[k[2]], color='black')
        ax.scatter(new_kx, new_ky, new_kz, marker='X', color='red', s=200)
        plt.show()
        break

    else:
        fig = plt.figure(figsize=(9 , 9))
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(sepal_length[k[0]], sepal_width[k[0]], petal_length[k[0]], color='yellow')
        ax.scatter(sepal_length[k[1]], sepal_width[k[1]], petal_length[k[1]], color='blue')
        ax.scatter(sepal_length[k[2]], sepal_width[k[2]], petal_length[k[2]], color='black')
        ax.scatter(new_kx, new_ky, new_kz, marker='X', color='purple', s=200)
        plt.show()
        kx , ky , kz = new_kx , new_ky , new_kz

