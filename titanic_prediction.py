import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv.xlsx.csv")

print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())
# Age column
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked column
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin column remove
df.drop("Cabin", axis=1, inplace=True)

print(df.isnull().sum())
df.drop(
    ["PassengerId", "Name", "Ticket"],
    axis=1,
    inplace=True
)

print(df.head())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])

df["Embarked"] = le.fit_transform(df["Embarked"])

print(df.head())
X = df.drop("Survived", axis=1)

y = df["Survived"]

print(X.head())

print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy =", accuracy)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "titanic_model.pkl")

print("Model Saved Successfully")