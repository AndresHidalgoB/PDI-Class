#Andres Felipe Hidalgo Betancourth 2205621
#Juan Camilo Hidalgo Betancourth 2205621

# Se importan las bibliotecas necesarias
import os#para interactuar con el sistema operativo y acceder a los archivos de imágenes
from tensorflow import keras #para cargar el modelo de red neuronal 
import numpy as np#manipular arreglos y matrices de datos
import matplotlib.pyplot as plt#mostrar las imágenes de prueba 
import cv2#para manipular las imágenes de prueba (cambiar tamaño, convertir a escala de grises)

model = keras.models.load_model("modeloConv.h5")#Se carga el modelo de red neuronal

etq = ["0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9"]#Se definen las etiquetas de los números del 0 al 9

# Se define una función para cargar las imágenes de prueba y preprocesarlas
def cargar_imgs():

  test_x = []  # Lista para almacenar las imágenes de prueba
  path = "Numeros"  # Ruta de la carpeta con las imágenes de prueba

  # Se recorren todas las imágenes de la carpeta
  for nom_archivo in os.listdir(path):
    
    img = cv2.imread(os.path.join(path, nom_archivo), cv2.IMREAD_GRAYSCALE)#Se carga cada imagen en escala de grises
    img = cv2.resize(img, (28, 28))#Se cambia el tamaño de la imagen a 28x28 (tamaño de entrada de la red neuronal)
    test_x.append(img)#Se agrega la imagen a la lista de imágenes de prueba

 
  test_x = np.array(test_x) #Se convierte la lista de imágenes de prueba en un arreglo NumPy
  test_x = test_x.reshape(-1, 28, 28, 1)#Se ajustan las dimensiones de las imágenes para que tengan el formato esperado por la red neuronal
  test_x = test_x.astype("float32") / 255#Se normalizan los valores de los pixeles de las imágenes

  return test_x

test_x = cargar_imgs()# Se cargan las imágenes de prueba
predicts = model.predict(test_x)#Se utilizan las imágenes de prueba para hacer predicciones con el modelo de red neuronal


for i in range(len(test_x)):#Se recorre cada imagen de prueba y su predicción correspondiente
  pred = np.argmax(predicts[i])#Se obtiene la predicción con mayor probabilidad utilizando la función argmax de NumPy
  etq_prediccion = etq[pred]#Se obtiene la etiqueta correspondiente a la predicción
  img = test_x[i].reshape(28, 28)#Se obtiene la imagen de prueba actual y se ajusta su forma para mostrarla
  

  plt.figure(figsize=(7,7))#Se define el tamaño de la figura de Matplotlib que muestra la imagen y la predicción
  plt.imshow(img, cmap="gray")#Se muestra la imagen de prueba 
  plt.title("Número identificado: " + etq_prediccion)#Se muestra la etiqueta de la predicción
  plt.axis("off")#Se elimina el eje de coordenadas de la figura
  plt.show()#Se muestra la figura de Matplotlib con la imagen y la predicción
  