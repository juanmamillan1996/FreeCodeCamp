import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Cargar los datos
df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# Revisar los datos
display(df.head())

# Convertir variables categóricas en variables numéricas
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_features = ['sex', 'smoker', 'region']
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Concatenar las variables numéricas y eliminar las originales
df = df.drop(columns=categorical_features)
df = pd.concat([df, categorical_df], axis=1)

# Separar características (X) y etiqueta (y)
X = df.drop(columns=['charges'])
y = df['charges']

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE (Error Absoluto Medio): {mae:.2f}')

# Visualizar los resultados
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicción de Costos de Salud")
plt.show()

# Verificar si cumple con el umbral de error
target_mae = 3500
if mae < target_mae:
    print("El modelo cumple con el criterio de precisión")
else:
    print("El modelo no cumple con el criterio de precisión, necesita ajustes")
