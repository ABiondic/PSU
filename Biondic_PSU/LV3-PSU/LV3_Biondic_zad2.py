import pandas as pd
import matplotlib.pyplot as plt


mtcars = pd.read_csv('mtcars.csv')


#1. zadatak

mpg_mean = mtcars.groupby('cyl')['mpg'].mean()

plt.figure()
mpg_mean.plot(kind='bar')
plt.title('Prosjecna potrosnja (mpg) po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Potrosnja (mpg)')
plt.show()



# 2. zadatak


mtcars.boxplot(column='wt', by='cyl')
plt.title('Distribucija tezine po cilindrima')
plt.suptitle('')
plt.xlabel('Broj cilindara')
plt.ylabel('Tezina')
plt.show()



# 3. zadatak


mtcars['transmission'] = mtcars['am'].map({0: 'Automatski', 1: 'Rucni'})


mtcars.boxplot(column='mpg', by='transmission')
plt.title('Potrosnja ovisno o mjenjacu')
plt.suptitle('')
plt.xlabel('Mjenjac')
plt.ylabel('Potrosnja (mpg)')
plt.show()





# 4. zadatak

plt.figure()

manual = mtcars[mtcars['am'] == 1]
auto = mtcars[mtcars['am'] == 0]

plt.scatter(manual['hp'], manual['qsec'], label='Rucni')
plt.scatter(auto['hp'], auto['qsec'], label='Automatski')

plt.title('Ubrzanje vs Snaga')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec)')
plt.legend()
plt.show()