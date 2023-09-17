from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

app = Flask(__name__)

# Cargar la estructura del modelo y sus pesos
with open("modelo.json", "r") as json_file:
    modelo_json = json_file.read()

modelo = model_from_json(modelo_json)
modelo.load_weights("modelo.h5")


@app.route('/predict', methods=['POST'])
def predict():
    # Comprobar si hay una imagen en la solicitud
    if 'file' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No image'}), 400

    try:
        # Procesar la imagen
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))  # Ajustar el tamaño según tu modelo
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.  # Si normalizaste tus imágenes durante el entrenamiento, asegúrate de hacerlo aquí también

         # Hacer predicción
        prediction = modelo.predict(image_array)

        # Obtener la clase predicha y su probabilidad
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[0][class_id])  # Probabilidad para la clase predicha

        response = {
            'prediction': class_id,
            'confidence': confidence
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
