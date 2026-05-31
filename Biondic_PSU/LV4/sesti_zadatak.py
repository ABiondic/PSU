# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error


# ucitavanje podataka
df = pd.read_csv("cars_processed (1).csv")

print(df.info())



df = df.drop(["name"], axis=1)


df = pd.get_dummies(df, drop_first=True)



X = df.drop("selling_price", axis=1)
y = df["selling_price"]

print("\nUlazne velicine nakon one-hot kodiranja:")
print(X.columns)

print("\nBroj ulaznih velicina nakon one-hot kodiranja:")
print(X.shape[1])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=300
)




Scaler = StandardScaler()

X_train_s = Scaler.fit_transform(X_train)
X_test_s = Scaler.transform(X_test)



linear_model = LinearRegression()
linear_model.fit(X_train_s, y_train)



y_pred_train = linear_model.predict(X_train_s)
y_pred_test = linear_model.predict(X_test_s)

print("\nRezultati na trening podacima:")
print("R2 train:", r2_score(y_train, y_pred_train))
print("MSE train:", mean_squared_error(y_train, y_pred_train))
print("RMSE train:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("MAE train:", mean_absolute_error(y_train, y_pred_train))
print("Max error train:", max_error(y_train, y_pred_train))

print("\nRezultati na testnim podacima:")
print("R2 test:", r2_score(y_test, y_pred_test))
print("MSE test:", mean_squared_error(y_test, y_pred_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("MAE test:", mean_absolute_error(y_test, y_pred_test))
print("Max error test:", max_error(y_test, y_pred_test))



y_pred_rupee = np.exp(y_pred_test)
y_test_rupee = np.exp(y_test)

print("\nPogreska u stvarnim cijenama:")
print("TRUE RMSE:", np.sqrt(mean_squared_error(y_test_rupee, y_pred_rupee)))
print("TRUE MAE:", mean_absolute_error(y_test_rupee, y_pred_rupee))


#Usporedba s 5.

df_old = pd.read_csv("cars_processed.csv")
df_old = df_old.drop(["name", "fuel", "seller_type", "transmission", "owner"], axis=1)

X_old = df_old.drop("selling_price", axis=1)
y_old = df_old["selling_price"]

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(
    X_old,
    y_old,
    test_size=0.2,
    random_state=300
)

Scaler_old = StandardScaler()
X_train_old_s = Scaler_old.fit_transform(X_train_old)
X_test_old_s = Scaler_old.transform(X_test_old)

linear_model_old = LinearRegression()
linear_model_old.fit(X_train_old_s, y_train_old)

y_pred_old = linear_model_old.predict(X_test_old_s)

print("\nUsporedba sa zadatkom 5, model samo s numerickim velicinama:")
print("R2 test:", r2_score(y_test_old, y_pred_old))
print("RMSE test:", np.sqrt(mean_squared_error(y_test_old, y_pred_old)))
print("MAE test:", mean_absolute_error(y_test_old, y_pred_old))

print("\nModel s numerickim i kategorickim velicinama:")
print("R2 test:", r2_score(y_test, y_pred_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("MAE test:", mean_absolute_error(y_test, y_pred_test))




fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_test, line_kws={"color": "green"})
ax.set(
    xlabel="Predikcija",
    ylabel="Stvarna vrijednost",
    title="Rezultati na testnim podacima s kategorickim varijablama"
)

plt.show()




#Komentar:
#U model su dodane kategoricke varijable pomocu funkcije pd.get_dummies.
#One-hot kodiranjem kategoricke varijable pretvorene su u numericke stupce.
#Rezultati se usporeduju s modelom iz zadatka 5 koji koristi samo numericke velicine.
#Dodavanje kategorickih varijabli obicno malo poboljsava rezultate.
#Ako se R2 poveca, a RMSE i MAE smanje, model je bolji.
#Poboljsanje nije nuzno veliko, ali kategoricke varijable ipak daju dodatne informacije o cijeni automobila.