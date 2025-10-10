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
    # Apple
    "Apple___Apple_scab": "Apply a fungicide containing captan or sulfur. Prune trees to improve air circulation.",
    "Apple___Black_rot": "Prune and dispose of infected branches. Apply a fungicide during the growing season.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees. Apply a preventative fungicide early in the season.",
    "Apple___healthy": "Your plant appears healthy. Continue with regular monitoring, proper watering, and fertilization.",

    # Blueberry
    "Blueberry___healthy": "Your plant appears healthy. Ensure acidic soil conditions and regular watering.",

    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": "Improve air circulation. Apply fungicides like sulfur or horticultural oil.",
    "Cherry_(including_sour)___healthy": "Your plant appears healthy. Ensure good soil drainage and continue regular care.",

    # Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Practice crop rotation and till the soil to bury residue. Apply a foliar fungicide if disease is severe.",
    "Corn_(maize)___Common_rust": "Plant resistant hybrids if available. Apply a foliar fungicide when rust is first detected.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids and practice crop rotation. Fungicide application may be necessary.",
    "Corn_(maize)___healthy": "Your plant appears healthy. Monitor for pests and ensure adequate nitrogen levels.",

    # Grape
    "Grape___Black_rot": "Apply fungicides starting early in the season. Remove and destroy infected vines and mummified berries.",
    "Grape___Esca_(Black_Measles)": "Prune infected canes during dormant season. No effective chemical control exists for established infections.",
    "Grape___Leaf_blight_(Isariopsis_leaf_spot)": "Improve air circulation. Fungicide sprays used for other diseases often control this.",
    "Grape___healthy": "Your plant appears healthy. Continue with proper pruning and pest management.",
    
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": "This is a serious disease with no cure. Remove the infected tree to prevent spread. Consult with local agricultural authorities immediately.",

    # Peach
    "Peach___Bacterial_spot": "Use resistant varieties. Apply copper-based bactericides during the dormant season.",
    "Peach___healthy": "Your plant appears healthy. Ensure proper pruning and thinning of fruit for best results.",

    # Pepper, Bell
    "Pepper,_bell___Bacterial_spot": "Plant disease-free seeds. Avoid overhead irrigation. Copper sprays can help manage spread.",
    "Pepper,_bell___healthy": "Your plant appears healthy. Ensure consistent watering and support for the plant as it grows.",

    # Potato
    "Potato___Early_blight": "Apply a preventative fungicide. Practice crop rotation and ensure good nutrition.",
    "Potato___Late_blight": "Apply fungicides regularly, especially in cool, moist conditions. Destroy infected plants immediately.",
    "Potato___healthy": "Your plant appears healthy. Monitor for potato beetles and ensure proper hilling of soil.",

    # Raspberry
    "Raspberry___healthy": "Your plant appears healthy. Prune canes after fruiting to promote new growth.",

    # Soybean
    "Soybean___healthy": "Your plant appears healthy. Monitor for aphids and other common pests.",
    
    # Squash
    "Squash___Powdery_mildew": "Improve air circulation. Apply fungicides like sulfur, potassium bicarbonate, or neem oil at the first sign of disease.",

    # Strawberry
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply a protective fungicide. Ensure good air circulation.",
    "Strawberry___healthy": "Your plant appears healthy. Protect from slugs and ensure consistent watering.",

    # Tomato
    "Tomato___Bacterial_spot": "Use copper-based bactericides. Avoid working with plants when they are wet.",
    "Tomato___Early_blight": "Apply a preventative fungicide. Mulch around the base of plants to prevent soil splash.",
    "Tomato___Late_blight": "Apply fungicides regularly. Ensure good air circulation and destroy infected plants.",
    "Tomato___Leaf_Mold": "Reduce humidity by improving ventilation. Fungicide application may be necessary.",
    "Tomato___Septoria_leaf_spot": "Remove and destroy infected lower leaves. Apply a fungicide containing chlorothalonil or mancozeb.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply insecticidal soap or horticultural oil. Introduce natural predators like ladybugs.",
    "Tomato___Target_Spot": "Practice crop rotation. Apply a preventative fungicide. Improve air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "This is a viral disease with no cure. Control whitefly populations to prevent spread. Remove and destroy infected plants.",
    "Tomato___Tomato_mosaic_virus": "This is a viral disease with no cure. Remove and destroy infected plants. Wash hands and tools to prevent spread.",
    "Tomato___healthy": "Your plant appears healthy. Continue to monitor for common pests like hornworms and aphids.",

    # Default message for uncertain predictions
    "Prediction Uncertain: Please upload a clearer image of a plant leaf.": "The system could not confidently identify the leaf. Please ensure the image is clear, well-lit, and shows a single leaf against a plain background."
}

# --- Configuration ---
app = Flask(__name__)
TARGET_SIZE = (224, 224) 
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static')
# CRUCIAL: Must match the file being created by train_model.py
WEIGHTS_FILE_PATH = 'smart_agri_weights_v2.weights.h5' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model Definition (Must match fine_tune_model.ipynb exactly) ---
NUM_CLASSES = 38

# 1. Load the base model
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False,
    input_shape=(224, 224, 3) 
)

# 2. Unfreeze the top layers for fine-tuning (THIS IS THE CRITICAL CHANGE)
base_model.trainable = True
fine_tune_at = 100 # We must freeze the same layers as during training
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 3. Define the Model using the Functional API structure
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
MODEL = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile the Model with the same low learning rate
MODEL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Use the fine-tuning learning rate
             loss='categorical_crossentropy', metrics=['accuracy'])


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

    # ADD THESE THREE LINES FOR DEBUGGING
print("---" * 20)
print(f"VERIFICATION: Successfully loaded model weights from: {WEIGHTS_FILE_PATH}")
print("---" * 20)
    
# 4. Define the 38 class names based on your dataset (CRUCIAL for output interpretation)
# This list must be in alphabetical order as determined by Keras during data loading
# In app.py, replace your entire CLASS_NAMES list with this one:

CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_leaf_spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
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
            
            # We are skipping the gatekeeper for now to focus on prediction accuracy
            label, confidence = model_predict(temp_path, MODEL)
            advice = TREATMENT_ADVICE.get(label, "Consult a local agricultural expert for specific treatment plans.")
            
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