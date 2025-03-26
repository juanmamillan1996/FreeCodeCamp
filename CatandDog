import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Definir algunas variables importantes
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Directorios de imágenes (estos deberían ajustarse según el entorno donde trabajes)
base_dir = "cats_and_dogs"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Generadores de imágenes para entrenar y validar
train_image_generator = ImageDataGenerator(rescale=1./255, 
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

validation_image_generator = ImageDataGenerator(rescale=1./255)

test_image_generator = ImageDataGenerator(rescale=1./255)

# Cargar imágenes desde los directorios
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=BATCH_SIZE,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

validation_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                     batch_size=BATCH_SIZE,
                                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                     class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                          batch_size=BATCH_SIZE,
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                          class_mode=None,  # No tiene etiquetas
                                                          shuffle=False)

# Definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Clasificación binaria (gato o perro)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
EPOCHS = 10  # Ajustable según rendimiento
history = model.fit(train_data_gen,
                    steps_per_epoch=len(train_data_gen),
                    epochs=EPOCHS,
                    validation_data=validation_data_gen,
                    validation_steps=len(validation_data_gen))

# Evaluar el modelo con imágenes de prueba
probabilities = model.predict(test_data_gen)
probabilities = [1 if prob > 0.5 else 0 for prob in probabilities]  # Convertir en etiquetas binarias (0=gato, 1=perro)

# Función para mostrar imágenes con sus predicciones
def plot_images(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(len(images)):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel("Dog" if labels[i] else "Cat")
    plt.show()

# Tomamos algunas imágenes del test para visualizarlas
sample_images, _ = next(test_data_gen)
plot_images(sample_images[:25], probabilities[:25])
