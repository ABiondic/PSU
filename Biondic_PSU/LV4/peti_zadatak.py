
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


df = df.drop(["name", "fuel", "seller_type", "transmission", "owner"], axis=1)

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

print("\nUlazne velicine:")
print(X.columns)




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

print("\nParametri modela:")
print("theta0 =", linear_model.intercept_)

for naziv, koef in zip(X.columns, linear_model.coef_):
    print(naziv, "=", koef)




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




print("\nPromjena broja ulaznih velicina:")

ulazne_velicine = [
    ["year", "km_driven"],
    ["year", "km_driven", "engine"],
    ["year", "km_driven", "engine", "max_power"],
    ["year", "km_driven", "mileage", "engine", "max_power", "seats"],
]

for velicine in ulazne_velicine:

    X = df[velicine]
    y = df["selling_price"]

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

    y_pred_test = linear_model.predict(X_test_s)

    print("\nUlazne velicine:", velicine)
    print("R2 test:", r2_score(y_test, y_pred_test))
    print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("MAE test:", mean_absolute_error(y_test, y_pred_test))



fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_test, line_kws={"color": "green"})
ax.set(
    xlabel="Predikcija",
    ylabel="Stvarna vrijednost",
    title="Rezultati na testnim podacima"
)

plt.show()


#Komentar_
#Model procjenjuje cijenu automobila na temelju numerickih ulaznih velicina.
#Kada se koristi manji broj ulaznih velicina, pogreska na testnom skupu je veca.
#Dodavanjem korisnih numerickih velicina model dobiva vise informacija, pa se pogreska uglavnom smanjuje.
#Ako neka dodatna velicina ne nosi puno korisnih informacija, poboljsanje moze biti malo.