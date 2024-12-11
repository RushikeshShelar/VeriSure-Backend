from tensorflow import keras
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def preprocess_document(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    return data

def classify_document(image_path):
    np.set_printoptions(suppress=True)

    model_path = "models/keras_model.h5"
    model = keras.models.load_model(model_path, compile=False)
    
    labels_path = "models/labels.txt"
    with open(labels_path, "r") as file:
        class_names = file.readlines()

    
    data = preprocess_document(image_path)
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    class_name = class_names[index].strip()
    class_name = class_name[2:]
    confidence_score = prediction[0][index]

    return class_name, confidence_score