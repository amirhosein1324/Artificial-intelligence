import numpy as np

engine_size = np.array([1.3 , 1.5 , 1.55 , 1.7 , 2 , 2.2 , 2.3 , 2.5 , 3 , 4])
fuel_consupsion = np.array([6 , 6.5 , 6.5 , 6.5 , 8 , 8.5 , 9 , 9.5 , 11 , 13])

x_bar = engine_size.mean()
y_bar = fuel_consupsion.mean()

a = np.sum(((engine_size - x_bar) * (fuel_consupsion - y_bar))) / np.sum((engine_size - x_bar) ** 2)
b = y_bar - (a * x_bar)

a = np.sum(((engine_size - x_bar) * (fuel_consupsion - y_bar))) / np.sum((engine_size - x_bar) ** 2)
b = y_bar - (a * x_bar)

x = 3.5
y = a * x + b
y

