import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)



df = pd.read_csv("occupancy_processed.csv")

print(df.head())
print(df.info())


X = df[['S3_Temp', 'S5_CO2']]
y = df['Room_Occupancy_Count']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


y_pred = knn.predict(X_test_scaled)

# a) Matrica zabune
from sklearn.metrics import confusion_matrix

mz = confusion_matrix(y_test, y_pred)

TN = mz[0, 0]
FP = mz[0, 1]
FN = mz[1, 0]
TP = mz[1, 1]

print("Matrica zabune:")
print(mz)

# b) Točnost
tocnost = (TP + TN) / (TP + TN + FP + FN)
print("Točnost:", tocnost)

# c) Preciznost
preciznost = TP / (TP + FP)
print("Preciznost:", preciznost)

# d) Odziv
odziv = TP / (TP + FN)
print("Odziv:", odziv)



print("\nUTJECAJ k:")

for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"k={k} -> tocnost={score:.4f}")




knn_no_scaling = KNeighborsClassifier(n_neighbors=5)
knn_no_scaling.fit(X_train, y_train)


X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

knn_no_scaling.fit(X_train_raw, y_train_raw)
y_pred_raw = knn_no_scaling.predict(X_test_raw)

print("\nTočnost BEZ skaliranja:")
print(accuracy_score(y_test_raw, y_pred_raw))