from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import json

def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def process_image(image_path, img_size=(28, 28)):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize(img_size, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def predict_image(image_path, model_path, class_names_path):
    model = load_model(model_path)

    class_names = load_class_names(class_names_path)

    processed_image = process_image(image_path)

    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)

    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

image_path = 'test.png'
model_path = 'resnet_quickdraw_model.h5'
class_names_path = 'class.json'

predicted_class = predict_image(image_path, model_path, class_names_path)
print(f"Predicted Class: {predicted_class}")
