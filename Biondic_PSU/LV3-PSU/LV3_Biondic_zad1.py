import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')


potrosnja=mtcars.sort_values(by=["mpg"])
gorivomax=potrosnja.head(5)
print(gorivomax.car)

cly_8 = mtcars[mtcars.cyl == 8]
sort_mpg_cly8 = cly_8.sort_values(by=['mpg'])
print(sort_mpg_cly8.tail(3))

cly_6 = mtcars[mtcars.cyl == 6]
print("Srednja vrijednost: ",cly_6['mpg'].mean())


tezina = mtcars[(mtcars.cyl == 4) & (mtcars.wt > 2.0) & (mtcars.wt < 2.2)]
print("Informacije o automobilu/ima s tezinom izmedu 2000 i 2200 lbs:")
print(tezina)


automatik = sum(mtcars.am==1)
mjenjac = sum(mtcars.am==0)
print("Broj automobila s automatskim mjenjacem je: ",automatik, ",a broj automobila s obicnim mjenjacem je: ",mjenjac)

autoistohp = ((mtcars['am'] == 1) & (mtcars['hp'] > 100)).sum()
print("Broj automobila s automatskim mjenjacem i da ima preko 100 hp-a: ",autoistohp)

