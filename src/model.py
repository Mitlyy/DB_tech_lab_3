import pandas as pd
import numpy as np
import os.path as pt
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys


main_path = pt.join(pt.dirname(__file__), "../")


df = pd.read_csv(pt.join(main_path, "data/data.csv"))
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)


X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].map({'M': 1, 'B': 0}) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class ConsoleProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        sys.stdout.write(f"\rЭпоха {epoch+1}: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}, val_loss={logs.get('val_loss'):.4f}, val_accuracy={logs.get('val_accuracy'):.4f}   ")
        sys.stdout.flush()

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0, callbacks=[ConsoleProgress()])
print()

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность модели: {accuracy:.4f}")

model.save(pt.join(main_path, "model/model.keras"))
joblib.dump(scaler, pt.join(main_path, "model/scaler.pkl"))

model = keras.models.load_model(pt.join(main_path, "model/model.keras"))
scaler = joblib.load(pt.join(main_path, "model/scaler.pkl"))
