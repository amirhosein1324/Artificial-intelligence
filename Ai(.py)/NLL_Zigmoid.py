import numpy as np

x = np.array([8 , 11 , 8 , 9 , 8 , 9 , 8 , 10])
y = np.array([1 , 1 , 0 , 0 , 0 , 1 , 1 , 0])
a = 1.5
b = 1.2
lr = 0.001
epsilon = 1e-3
mean = np.mean(x)
mean

std = np.std(x, ddof=1)
std

normalized_x = (x - mean) / std
normalized_x


n = 5
K = 0
previous_error = 0

for epoch in range(10000):
  z = a * normalized_x + b
  p = 1 / (1 + np.exp(-z))
  grad_a = np.sum((p-y) * normalized_x)
  grad_b = np.sum(p-y)
  a -= lr * grad_a
  b -= lr * grad_b
  error_formula = -y * (np.log2 (p) ) - (1-y) * (np.log2 (1-p) )
  error = np.mean(error_formula)

  if abs(previous_error - error) > epsilon:
    previous_error = error
    K = 0
  else:
    K += 1
  if K == n:
    print(f"after {epoch} epochs")
    print(f"result a: {a}")
    print(f"result b: {b}")
    print(f"error {error}")
    break

