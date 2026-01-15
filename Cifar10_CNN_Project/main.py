import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("1. Veri seti yükleniyor...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

class_names = ['Uçak', 'Otomobil', 'Kuş', 'Kedi', 'Geyik', 
               'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

unique, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(8, 4))
plt.bar(class_names, counts, color='teal')
plt.title("Eğitim Seti (Devam etmek için pencereyi KAPATIN)")
plt.show()

X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Eğitim başlıyor...")
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Test Başarısı')
plt.xlabel('Epoch')
plt.ylabel('Başarı')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()