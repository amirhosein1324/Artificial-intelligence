import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 10, 0.5)
y = 2 * x + 3

plt.plot(x,y)

x = np.linspace(-2 * np.pi , 2 * np.pi , 10)
y = np.sin(x)
plt.plot(x,y)

x = np.linspace(-2 * np.pi , 2 * np.pi , 100)
y = np.sin(x)
plt.plot(x,y)

income = np.array([100 , 1500 , 1500 , 1800 , 2000 , 2000 , 2000 , 3000 , 3000 , 10000])
age = np.array([85 , 33 , 32 , 35 , 10 , 45 , 28 , 40 , 38 , 50])
plt.scatter(income , age)

plt.hist(income , bins = 5)

x = np.random.normal(100 , 15 , 1000000)
plt.hist(x , bins = 100)

import seaborn as sns
sns.histplot(x , kde = True , color = "black")

# 1 تمرین :  هزار عدد بین 10 و - 10 تولید بشه که روی ایگرگ نویز ایجاد بشه با میانگین 0 و انحراف معیار
plt.figure(figsize =(10 , 10))
x = np.linspace(-10 , 10 , 1000)
y = x * 3 + 5
meow = plt.scatter(x , y)
y2 =  np.random.normal(0 , 1 , 1000)
y = y + y2
nigga = plt.scatter(x , y)

plt.figure(figsize =(15 , 15))
x = np.linspace(-10 , 10 , 1000)
y = x * 3 + 5
meow = plt.scatter(x , y)
y2 =  np.random.normal(0 , 3 , 1000)
y = y + y2
nigga = plt.scatter(x , y)

plt.figure(figsize =(15 , 15))
x = np.linspace(-10 , 10 , 1000)
y = x * 3 + 5
meow = plt.scatter(x , y)
y2 =  np.random.normal(0 , 3 , 1000)
y_hat = y + y2
nigga = plt.scatter(x , y)
plt.plot(x , y , label = "plot_meow")
plt.scatter(x,y ,label = "plt_nazi")
plt.xlabel("count")
plt.ylabel("label 3")
plt.title("PLOT")
plt.legend()

plt.figure(figsize =(15 , 15))
x = np.linspace(-10 , 10 , 1000)
y = x * 3 + 5
meow = plt.scatter(x , y)
y_hat =  np.random.normal(0 , 1 , 1000)
y = y + y_hat
nigga = plt.scatter(x , y)

plt.plot(x , y , label = "plot_one")
plt.scatter(x,y ,label = "plt_two")
plt.xlabel("range")
plt.ylabel("count")
plt.title("PLOT")
plt.legend()

plt.figure(figsize =(20 , 20))
x = np.linspace(-10 , 10 , 1000)
y = x * 3 + 5
s = plt.scatter(x , y , label = "one")
n =  np.random.normal(0 , 2 , 1000)
y_hat = y + n
s_hat = plt.scatter(x , y_hat , c = "r" , label = "two")
plt.title("PLOT")
plt.legend()

# Fit Regression for up plot , y = ax + b
a  = np.sum((x - np.mean(x)) * (y_hat - np.mean(y_hat))) / np.sum((x - np.mean(x)) ** 2)
print(f"a : {a}")
b = np.mean(y_hat) - a * np.mean(x)
print(f"b : {b}")



