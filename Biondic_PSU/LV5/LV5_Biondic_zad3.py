import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score


df = pd.read_csv("occupancy_processed.csv")

print(df.head())
print(df.info())


X = df[['S3_Temp', 'S5_CO2']]
y = df['Room_Occupancy_Count']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)


mz = confusion_matrix(y_test, y_pred)

print("\nMATRICA ZABUNE:")
print(mz)


TN = mz[0, 0]
FP = mz[0, 1]
FN = mz[1, 0]
TP = mz[1, 1]

print("\nTP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)



# točnost
tocnost = (TP + TN) / (TP + TN + FP + FN)
print("\nTočnost:", tocnost)

# preciznost
preciznost = TP / (TP + FP)
print("Preciznost:", preciznost)

# odziv
odziv = TP / (TP + FN)
print("Odziv:", odziv)


plt.figure(figsize=(15, 8))

plot_tree(
    tree,
    feature_names=['S3_Temp', 'S5_CO2'],
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True
)

plt.title("Stablo odlučivanja")
plt.show()


print("\nUTJECAJ max_depth:")

for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"max_depth={depth} -> točnost={score:.4f}")



X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

tree_raw = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_raw.fit(X_train_raw, y_train_raw)

y_pred_raw = tree_raw.predict(X_test_raw)

print("\nTočnost BEZ skaliranja:")
print(accuracy_score(y_test_raw, y_pred_raw))