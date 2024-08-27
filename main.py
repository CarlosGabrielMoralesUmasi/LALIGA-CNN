
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('LaLiga_dataset.csv')

# Seleccionar las características y la etiqueta
features = df[['home_goals', 'away_goals', 'goal_difference']].values
labels = df['points'].values

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Convertir etiquetas a una escala similar
labels = labels.reshape(-1, 1)
scaled_labels = scaler.fit_transform(labels)

# Dividir en secuencias para la RNN
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(scaled_features, scaled_labels, seq_length)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir el modelo LSTM con regularización y dropout ajustado
model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer='l2')))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(units=50, kernel_regularizer='l2')))
model.add(Dropout(0.4))
model.add(Dense(units=1))

# Compilar el modelo con una tasa de aprendizaje reducida
model.compile(optimizer='adam', loss='mean_squared_error')

# Implementar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Hacer predicciones
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Graficar los resultados
plt.figure(figsize=(14, 5))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Real')
plt.plot(predictions, color='red', label='Predicciones')
plt.title('Predicción de los Puntos del Equipo')
plt.xlabel('Equipos')
plt.ylabel('Puntos')
plt.legend()
plt.show()
