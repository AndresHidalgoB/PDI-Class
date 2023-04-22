#Andres Felipe Hidalgo Betancourth 2205621
#Juan Camilo Hidalgo Betancourth 2205621

import cv2
import numpy as np
import os

a = "C:/Users/andre/Pictures/Frutas/" #Ruta donde estan las imagenes .jpg

def find_fruit_contours(path):
    # Definimos los limites del rango de colores para la deteccion de verdes y rojos
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86,255,255])
    lower_green2 = np.array([25, 52, 72])
    upper_green2 = np.array([102, 255, 255])

    # Recorre la ruta y encuentra los contornos de las frutas
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carga la imagen
            image = cv2.imread(os.path.join(path, filename))
            #Convierte a HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # creamos las máscaras para obtener solo los píxeles dentro del rango de color
            mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
            mask_green_final = cv2.bitwise_or(cv2.bitwise_and(mask_green, mask_green2), mask_green2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask = cv2.bitwise_or(mask_red, mask_green_final)

            # Aplica un filtro Gaussiano para suavizar los bordes de la imagen
            blurred = cv2.GaussianBlur(mask, (11, 11), 0)

            # Encuentra los contornos de las frutas en la imagen binaria
            contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dibuja los contornos en la imagen original
            image_contours = image.copy()
            cv2.drawContours(image_contours, contours, -1, (214, 255, 0), 2)

            # Muestra la imagen con los contornos dibujados
            cv2.imshow("Contornos de frutas", image_contours)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

find_fruit_contours(a)




