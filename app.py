import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
# We use the Model architecture, not load_model, to avoid serialization errors
import tensorflow as tf 
from tensorflow.keras.applications import MobileNetV2 # Import the correct base model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing.image import img_to_array

# --- Configuration ---
app = Flask(__name__)
TARGET_SIZE = (224, 224) 
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static')
# CRUCIAL: Must match the file being created by train_model.py
WEIGHTS_FILE_PATH = 'smart_agri_weights.weights.h5' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model Definition (Must match train_model.py exactly) ---

NUM_CLASSES = 38 

# 1. Define the MobileNetV2 Base Model 
try:
    # Forces Keras to download/use the full MobileNetV2 weights from ImageNet
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False,
        input_shape=(224, 224, 3) 
    )
    base_model.trainable = False

    # 2. Define the Model using the Functional API structure
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    MODEL = Model(inputs=base_model.input, outputs=predictions)

    # 3. Compile the Model (Required before loading weights)
    MODEL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                 loss='categorical_crossentropy', metrics=['accuracy'])

except Exception as e:
    print(f"FATAL ERROR during model definition: {e}")
    exit()

# --- Load Trained Weights (Guaranteed Method) ---

try:
    # Load weights directly into the defined model structure
    MODEL.load_weights(WEIGHTS_FILE_PATH)
    print("AI Model weights loaded successfully! System operational.")
except Exception as e:
    # This error means the file is not finished saving yet or does not exist
    print(f"FATAL ERROR loading model weights: {e}")
    print(f"Please check that '{WEIGHTS_FILE_PATH}' exists in your project folder and training is complete.")
    exit() 
    
# 4. Define the 38 class names based on your dataset (CRUCIAL for output interpretation)
# This list must be in alphabetical order as determined by Keras during data loading
CLASS_NAMES = [
    'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Apple___scab', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_leaf_spot)', 
    'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___healthy', 'Background_without_leaves', 'Blueberry___healthy'
]
    
# --- Prediction Function ---
def model_predict(image_path, model):
    """Loads, preprocesses, and predicts the class of an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        
        x = img_to_array(img)
        x = x / 255.0  # Normalize
        x = np.expand_dims(x, axis=0) # Add batch dimension

        prediction = model.predict(x)[0]
        
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]
        predicted_label = CLASS_NAMES[predicted_index]
        
        return predicted_label, confidence
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Prediction Failed", 0.0

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handles file uploads and displays the diagnosis result."""
    prediction_data = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            
            # Save the file first, then run prediction
            file.save(temp_path)
            
            label, confidence = model_predict(temp_path, MODEL)
            
            prediction_data = {
                'label': label,
                'confidence': float(confidence)
            }
            
    return render_template('index.html', prediction=prediction_data)

# --- Run the App ---
if __name__ == '__main__':
    # Make sure the static folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        
    app.run(debug=True)