import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

def preprocess_rgb_images(image, sigmaX=10):
    try:
        image = load_ben_color(image, sigmaX=30)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_imagen(image):
    try:
        model = load_model('models/BestModel-densenet121_preprocessed-retinal.h5')
        predictions = model.predict(image)
        predicted_class = 1 if predictions[0] >= 0.5 else 0
        class_names = ["Sano", "Enfermo"]
        return class_names[predicted_class]
    except Exception as e:
        print(f"Error al realizar la predicción: {e}")
        return None

def load_model(model_path):
    try:
        print("Intentando cargar el modelo desde:", model_path)
        model = tf.keras.models.load_model(model_path)
        print("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print("Error al cargar el modelo:", e)
        return None
