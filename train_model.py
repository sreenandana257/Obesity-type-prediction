import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


df = pd.read_csv("Desktop\mlai\Assignment ANN with Web Interface\ObesityDataSet_raw_and_data_sinthetic.csv")

print(df.head())
print(df.info())


print(df["NObeyesdad"].value_counts())



plt.figure(figsize=(10,6))
df["NObeyesdad"].value_counts().plot(kind='bar')
plt.title("Obesity Level Distribution")
plt.xlabel("Obesity Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("obesity_distribution.png")

plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_distribution.png")

plt.figure(figsize=(8,5))
plt.hist(df["Weight"], bins=20)
plt.title("Weight Distribution")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.savefig("weight_distribution.png")

plt.figure(figsize=(8,6))

categories = df["NObeyesdad"].unique()
colors = ['red','blue','green','orange','purple','brown','cyan']

for cat, color in zip(categories, colors):
    subset = df[df["NObeyesdad"] == cat]
    plt.scatter(subset["Height"],
                subset["Weight"],
                label=cat,
                color=color)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight")
plt.legend()
plt.savefig("plot.png")

encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)


model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)


pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

acc = accuracy_score(y_test, pred)
print("Accuracy =", acc)

model.save("obesity_ann_model.h5")


joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Saved successfully.")
