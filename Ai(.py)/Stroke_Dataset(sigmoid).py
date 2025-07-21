import numpy as np

dataset = np.loadtxt('Stroke.csv' , delimiter="," , dtype = str)
dataset

dataset[0]

# bmi None before deleting = 5111
stroke_y = dataset[1:, [2, 8, 9, -1]]
stroke_y = np.delete(stroke_y, np.where(stroke_y == "N/A")[0], axis=0)
print(len(stroke_y))

stroke_df = stroke_y.astype(dtype=np.float16)
X = stroke_df[:, :3]
Y = stroke_df[:, -1]
age_normal = np.mean(X[:, 0]) / np.std(X[:, 0])
avg_gl_normal = np.mean(X[:, 1]) / np.std(X[:, 1])
bmi_normal = np.mean(X[:, 2]) / np.std(X[:, 2])

a = 1
b = 1
c = 1
d = 1
lr = 0.001
epsilon = 1e-5
best_error = 0
epochs_list = []

early_stop = len(X) * 20 // 100
k = 0
for epoch in range(100000):
    z = a * age_normal + b * avg_gl_normal + c * bmi_normal + d
    p = 1 / (1 + np.exp(-z))

    a_grad = np.sum((p - Y) * age_normal)
    b_grad = np.sum((p - Y) * avg_gl_normal)
    c_grad = np.sum((p - Y) * bmi_normal)
    d_grad = np.sum(p - Y)

    a -= a_grad * lr
    b -= b_grad * lr
    c -= c_grad * lr
    d -= d_grad * lr

    error = np.mean(-Y * np.log2(p) - (1 - Y) * np.log2(1 - p))
    print(error)
    epochs_list.append(epoch)

    if abs(best_error - error) > epsilon:
        best_error = error
        k = 0
    else:
        k += 1
    if k == early_stop:
        print(f"after {epoch} epochs.")
        print(f"final a: {a}")
        print(f"final b: {b}")
        print(f"final c: {c}")
        print(f"final d: {d}")
        print(f"error: {error}")
        break



