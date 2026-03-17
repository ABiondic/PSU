import numpy as np

import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

mpg= data[:,0]
hp= data[:,3]
wt= data[:,5]
cyl = data[:,1]

plt.scatter(mpg,hp,s=wt,color='blue',label='Group 1')
plt.xlabel('MPG')
plt.ylabel('HP')
plt.title('Graf informacija automobila')
plt.legend()
plt.show()

mpg6=mpg[cyl==6]

print(mpg.min())
print(mpg.max())
print(mpg.mean())
print("\n")
print(mpg6.min())
print(mpg6.max())
print(mpg6.mean())



