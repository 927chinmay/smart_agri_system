import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
# We use the Model architecture, not load_model, to avoid serialization errors
import tensorflow as tf 
from tensorflow.keras.applications import MobileNetV2 # Import the correct base model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing.image import img_to_array



# --- Treatment Recommendations Data ---
TREATMENT_ADVICE = {
    "Apple___Apple_scab": "Apply a fungicide containing captan or sulfur. Ensure good air circulation by pruning trees.",
    "Apple___Black_rot": "Prune and dispose of infected branches and mummified fruit. Apply a fungicide during the growing season.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees if possible. Apply a preventative fungicide from pink-bud stage until fruits are mature.",
    "Apple___healthy": "Your plant appears healthy. Continue with regular monitoring, proper watering, and fertilization.",
    "Cherry_(including_sour)___Powdery_mildew": "Improve air circulation. Apply fungicides like sulfur, potassium bicarbonate, or horticultural oil.",
    "Cherry_(including_sour)___healthy": "Your plant appears healthy. Ensure good soil drainage and continue regular care.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Practice crop rotation and till the soil to bury residue. Apply a foliar fungicide if disease is severe.",
    "Corn_(maize)___healthy": "Your plant appears healthy. Monitor for pests and ensure adequate nitrogen levels.",
    "Tomato___Bacterial_spot": "Avoid overhead watering. Use copper-based bactericides as a preventative measure. Remove and destroy infected plants.",
    "Tomato___Leaf_Mold": "Ensure proper spacing and ventilation to reduce humidity. Apply a fungicide if necessary.",
    "Tomato___healthy": "Your plant appears healthy. Continue to monitor for common pests like hornworms and aphids.",
    # This is the default message for uncertain predictions
    "Prediction Uncertain: Please upload a clearer image of a plant leaf.": "The system could not confidently identify the leaf. Please ensure the image is clear, well-lit, and shows a single leaf against a plain background."
}

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

# app.py (Paste this new function after your CLASS_NAMES list, around line 118)
def is_likely_plant_leaf(image_path, green_threshold=0.15):
    """
    Checks if an image is likely a plant leaf based on the percentage of green color.
    """
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        # Convert image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for the color green in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create a mask that captures only the green pixels
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Calculate the percentage of green pixels
        green_percentage = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
        
        print(f"Detected green percentage: {green_percentage:.2f}")

        # If green percentage is above our threshold, it's likely a leaf
        return green_percentage > green_threshold
    except Exception as e:
        print(f"Color analysis error: {e}")
        return False
    
# --- Prediction Function ---
# app.py

# --- Define the threshold for accepting a prediction (85% is safe given our 96.67% accuracy) ---
CONFIDENCE_THRESHOLD = 0.85

def model_predict(image_path, model):
    """Loads, preprocesses, and predicts the class of an image with an OOD check."""
    try:
        # (Standard preprocessing steps remain the same)
        img = Image.open(image_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        
        x = img_to_array(img)
        x = x / 255.0 
        x = np.expand_dims(x, axis=0) 

        # Make the prediction
        prediction = model.predict(x)[0]
        
        # Find the highest confidence and its corresponding label
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]
        predicted_label = CLASS_NAMES[predicted_index]
        
        # --- OOD DETECTION LOGIC ---
        # Check if the confidence is below our threshold
        if confidence < CONFIDENCE_THRESHOLD:
            # If the model is not confident, reject the prediction.
            # We return a specific label and the low confidence score for display.
            return "Prediction Uncertain: Please upload a clearer image of a plant leaf.", confidence
        # --- END OOD LOGIC ---
        
        return predicted_label, confidence
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Prediction Failed", 0.0
    
# --- Flask Routes ---
# app.py (Replace your existing upload_file function with this one)

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
            file.save(temp_path)
            
            # --- GATEKEEPER LOGIC ---
            # First, check if the image is likely a plant leaf
            if not is_likely_plant_leaf(temp_path):
                label = "Invalid Image"
                confidence = 1.0 # Set confidence to 100% for the invalid message
                advice = "This does not appear to be a plant leaf. Please upload a clear image of a single leaf."
            else:
                # If it passes the gatekeeper, proceed with disease prediction
                label, confidence = model_predict(temp_path, MODEL)
                advice = TREATMENT_ADVICE.get(label, "Consult a local agricultural expert for specific treatment plans.")
            # --- END GATEKEEPER LOGIC ---
            
            prediction_data = {
                'label': label,
                'confidence': float(confidence),
                'advice': advice
            }
            
    return render_template('index.html', prediction=prediction_data)

# --- Run the App ---
if __name__ == '__main__':
    # Make sure the static folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        
    app.run(debug=True)