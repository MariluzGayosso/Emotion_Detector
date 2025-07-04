import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Cargar el dataset
data = pd.read_csv('dataset/fer2013.csv')

# Separar características y etiquetas
pixels = data['pixels'].tolist()
faces = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48) for pixel in pixels])
faces = faces.astype('float32') / 255.0
faces = np.expand_dims(faces, -1)
emotions = to_categorical(data['emotion'], num_classes=7)

# Dividir en entrenamiento y prueba (90/10)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)

# Crear el modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emociones
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Guardar modelo
os.makedirs('modelos', exist_ok=True)
model.save('modelos/modelo_emociones.hdf5')
