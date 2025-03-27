# Primero, importamos las librerías necesarias para el proyecto
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargamos el dataset desde una URL
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms_spam.csv"
df = pd.read_csv(url, encoding="latin-1")

# Eliminamos las columnas que no necesitamos, dejando solo las etiquetas y los mensajes
df = df[['v1', 'v2']]  # v1 es 'ham' o 'spam', v2 es el mensaje real

# Ahora, mapeamos las etiquetas 'ham' y 'spam' a números: 0 para 'ham' y 1 para 'spam'
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

# Dividimos los datos en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

# Usamos CountVectorizer para convertir los mensajes en vectores numéricos
# Esto permite que el modelo pueda trabajar con el texto de manera efectiva
vectorizer = CountVectorizer(stop_words='english')  # Excluimos palabras comunes que no aportan mucha información
X_train_vec = vectorizer.fit_transform(X_train)  # Ajustamos y transformamos los datos de entrenamiento
X_test_vec = vectorizer.transform(X_test)  # Transformamos los datos de prueba

# Ahora creamos el modelo de red neuronal
# Vamos a hacer una red sencilla con 3 capas
model = Sequential()
model.add(Dense(512, input_dim=X_train_vec.shape[1], activation='relu'))  # Capa de entrada con 512 neuronas
model.add(Dense(256, activation='relu'))  # Capa intermedia con 256 neuronas
model.add(Dense(1, activation='sigmoid'))  # Capa de salida: 1 neurona con activación 'sigmoid' para clasificación binaria

# Compilamos el modelo, usando 'binary_crossentropy' como función de pérdida (porque es clasificación binaria)
# Usamos el optimizador 'adam', que es muy popular para estos casos
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamos el modelo con los datos de entrenamiento
# Aquí le decimos que usemos 5 épocas y un tamaño de lote de 64
model.fit(X_train_vec, y_train, epochs=5, batch_size=64, validation_data=(X_test_vec, y_test))

# Evaluamos el modelo con los datos de prueba para ver qué tan bien lo hemos entrenado
loss, accuracy = model.evaluate(X_test_vec, y_test)
print(f"El modelo ha sido evaluado. Exactitud: {accuracy*100:.2f}%")

# Creamos una función para predecir si un mensaje es 'ham' o 'spam'
# Esta función toma un mensaje, lo convierte en un vector y luego lo pasa al modelo
def predict_message(message):
    message_vec = vectorizer.transform([message])  # Convertimos el mensaje en un vector
    prediction = model.predict(message_vec)  # Hacemos la predicción
    
    # Si la predicción es mayor o igual a 0.5, lo consideramos 'spam'
    if prediction >= 0.5:
        return [prediction[0][0], 'spam']
    else:
        return [prediction[0][0], 'ham']

# Probamos con un mensaje de ejemplo para ver si nuestra función funciona bien
test_message = "Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005"
result = predict_message(test_message)
print(f"Predicción para el mensaje: {result}")
