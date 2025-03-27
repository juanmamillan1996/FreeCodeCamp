import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Cargar los datos de los archivos CSV
print("Cargando datos...")
books = pd.read_csv("BX-Books.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
users = pd.read_csv("BX-Users.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)

# Renombrar columnas para facilitar el manejo
i
books.columns = ["ISBN", "Title", "Author", "Year", "Publisher", "ImageURL_S", "ImageURL_M", "ImageURL_L"]
ratings.columns = ["UserID", "ISBN", "Rating"]
users.columns = ["UserID", "Location", "Age"]

# Filtrar los datos para obtener usuarios y libros con suficientes calificaciones
print("Filtrando datos...")
user_counts = ratings["UserID"].value_counts()
book_counts = ratings["ISBN"].value_counts()

# Mantener solo usuarios con más de 200 calificaciones y libros con más de 100 calificaciones
valid_users = user_counts[user_counts > 200].index
valid_books = book_counts[book_counts > 100].index

ratings_filtered = ratings[(ratings["UserID"].isin(valid_users)) & (ratings["ISBN"].isin(valid_books))]

# Crear matriz de utilidad (usuarios vs libros)
print("Creando matriz de utilidad...")
book_pivot = ratings_filtered.pivot(index="ISBN", columns="UserID", values="Rating").fillna(0)
matrix = csr_matrix(book_pivot.values)

# Entrenar modelo KNN para recomendaciones
print("Entrenando modelo KNN...")
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(matrix)

# Función para obtener recomendaciones
def get_recommends(book_title):
    print(f"Buscando recomendaciones para: {book_title}")
    
    # Verificar si el libro existe en el dataset
    if book_title not in books["Title"].values:
        print("Libro no encontrado en el dataset.")
        return None
    
    # Obtener el ISBN del libro
    book_isbn = books[books["Title"] == book_title]["ISBN"].values[0]
    
    # Verificar si el libro tiene suficientes calificaciones
    if book_isbn not in book_pivot.index:
        print("Este libro no tiene suficientes calificaciones para hacer recomendaciones.")
        return None
    
    # Obtener índice del libro en la matriz
    book_idx = book_pivot.index.get_loc(book_isbn)
    
    # Encontrar los libros más cercanos
    distances, indices = model.kneighbors(matrix[book_idx], n_neighbors=6)  # 6 porque el primer resultado será el mismo libro
    
    # Obtener títulos de los libros recomendados
    recommended_books = []
    for i in range(1, len(indices[0])):  # Saltamos el primer índice porque es el mismo libro
        isbn_rec = book_pivot.index[indices[0][i]]
        title_rec = books[books["ISBN"] == isbn_rec]["Title"].values[0]
        recommended_books.append([title_rec, distances[0][i]])
    
    return [book_title, recommended_books]

# Prueba de la función
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
