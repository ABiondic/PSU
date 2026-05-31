import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ucitavanje ociscenih podataka
df = pd.read_csv("cars_processed (1).csv")

# kopija originalnog dataframea za odgovore na pitanja
df_original = df.copy()

print(df.info())

# razliciti prikazi
sns.pairplot(df, hue="fuel")
sns.relplot(data=df, x="km_driven", y="selling_price", hue="fuel")

df = df.drop(["name", "mileage"], axis=1)


obj_cols = df.select_dtypes(include=["object", "str"]).columns.values.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.values.tolist()

fig = plt.figure(figsize=[15, 8])
for col in range(len(obj_cols)):
    plt.subplot(2, 2, col + 1)
    sns.countplot(x=obj_cols[col], data=df)

df.boxplot(by="fuel", column=["selling_price"], grid=False)

df.hist(["selling_price"], grid=False)


tabcorr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(tabcorr, annot=True, linewidths=2, cmap="coolwarm")



print("\n1. Koliko mjerenja (automobila) je dostupno u datasetu?")
print("Dostupno je", df_original.shape[0], "automobila.")

print("\n2. Kakav je tip pojedinog stupca u dataframeu?")
print(df_original.dtypes)

print("\n3. Koji automobil ima najvecu cijenu, a koji najmanju?")
najveca_cijena = df_original.loc[df_original["selling_price"].idxmax()]
najmanja_cijena = df_original.loc[df_original["selling_price"].idxmin()]

print("Najvecu cijenu ima:", najveca_cijena["name"])
print("Najveca log cijena:", najveca_cijena["selling_price"])

print("Najmanju cijenu ima:", najmanja_cijena["name"])
print("Najmanja log cijena:", najmanja_cijena["selling_price"])

print("\n4. Koliko automobila je proizvedeno 2012. godine?")
print("Broj automobila proizvedenih 2012. godine:", (df_original["year"] == 2012).sum())

print("\n5. Koji automobil je presao najvise kilometara, a koji najmanje?")
najvise_km = df_original.loc[df_original["km_driven"].idxmax()]
najmanje_km = df_original.loc[df_original["km_driven"].idxmin()]

print("Najvise kilometara je presao:", najvise_km["name"])
print("Kilometri:", najvise_km["km_driven"])

print("Najmanje kilometara je presao:", najmanje_km["name"])
print("Kilometri:", najmanje_km["km_driven"])

print("\n6. Koliko najcesce automobili imaju sjedala?")
print("Najcesci broj sjedala:", df_original["seats"].mode()[0])

print("\n7. Kolika je prosjecna prijedena kilometraza za Diesel i Petrol automobile?")
print("Diesel:", df_original[df_original["fuel"] == "Diesel"]["km_driven"].mean())
print("Petrol:", df_original[df_original["fuel"] == "Petrol"]["km_driven"].mean())

plt.show()