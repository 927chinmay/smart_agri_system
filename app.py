import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, g
# We use the Model architecture
import tensorflow as tf 
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing.image import img_to_array
import datetime # NEW: Added for timestamps
import sqlite3 # NEW: Added for the database

# --- Configuration ---
app = Flask(__name__)
TARGET_SIZE = (224, 224) 
# UPDATED: Set a single, correct upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads' 
WEIGHTS_FILE_PATH = 'smart_agri_weights_v2.weights.h5' 

# --- Database Configuration ---
DATABASE = 'history.db'

def get_db():
    """Connects to the SQLite database."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # Allows us to access columns by name
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Closes the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the database and creates the 'predictions' table if it doesn't exist."""
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# --- Treatment Recommendations Data ---
TREATMENT_ADVICE = {
    # (Your full dictionary is perfect, no changes needed)
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
    # Default message 
    "Prediction Uncertain: Please upload a clearer image of a plant leaf.": "The system could not confidently identify the leaf. Please ensure the image is clear, well-lit, and shows a single leaf against a plain background."
}

#  Model Definition 
NUM_CLASSES = 38
# (Your model definition is perfect, no changes needed)
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False,
    input_shape=(224, 224, 3) 
)
base_model.trainable = True
fine_tune_at = 100 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
MODEL = Model(inputs=base_model.input, outputs=predictions)
MODEL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
             loss='categorical_crossentropy', metrics=['accuracy'])

# --- Load Trained Weights ---
try:
    MODEL.load_weights(WEIGHTS_FILE_PATH)
    print("AI Model weights loaded successfully! System operational.")
except Exception as e:
    print(f"FATAL ERROR loading model weights: {e}")
    print(f"Please check that '{WEIGHTS_FILE_PATH}' exists in your project folder and training is complete.")
    exit() 
    
print("---" * 20)
print(f"VERIFICATION: Successfully loaded model weights from: {WEIGHTS_FILE_PATH}")
print("---" * 20)
    
# --- Classnames ---
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

# REMOVED: Deleted the unused `is_likely_plant_leaf` function to clean up the code.

# --- Prediction Function ---
CONFIDENCE_THRESHOLD = 0.70 # Set to 70% as we discussed

def model_predict(image_path, model):
    """Loads, preprocesses, and predicts the class of an image with an OOD check."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        
        x = img_to_array(img)
        x = x / 255.0 
        x = np.expand_dims(x, axis=0) 

        prediction = model.predict(x)[0]
        
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]
        
        # --- OOD DETECTION LOGIC ---
        if confidence < CONFIDENCE_THRESHOLD:
            # Return a special index (-1) for uncertainty
            return "Prediction Uncertain: Please upload a clearer image of a plant leaf.", confidence, -1
        # --- END OOD LOGIC ---
        
        predicted_label = CLASS_NAMES[predicted_index]
        # Return the label, confidence, AND the index
        return predicted_label, confidence, predicted_index
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Prediction Failed", 0.0, -1
    
    # --- NEW: Grad-CAM Explainability Function ---
def get_gradcam_heatmap(image_path, predicted_index):
    """Generates a Grad-CAM heatmap overlay for the given image and prediction."""
    try:
        # 1. Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # 2. Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(MODEL.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            print("Could not find a Conv2D layer.")
            return None

        # 3. Create a sub-model
        grad_model = Model(
            [MODEL.inputs], [last_conv_layer.output, MODEL.output]
        )

        # 4. Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, predicted_index]
        
        grads = tape.gradient(loss, conv_outputs)
        
        # 5. Pool the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 6. Generate the heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) # Apply ReLU
        heatmap /= np.max(heatmap)
        heatmap = heatmap.squeeze()

        # 7. Superimpose heatmap on original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, TARGET_SIZE)
        
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap_color * 0.4 + original_img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        # 8. Save the heatmap image
        heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        cv2.imwrite(heatmap_path, superimposed_img)
        
        # Return the URL for the template
        heatmap_url = url_for('static', filename='uploads/' + heatmap_filename)
        return heatmap_url

    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None
    
# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handles file uploads, runs prediction, and generates Grad-CAM heatmap."""
    prediction_data = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save file with a unique name
            filename = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{file.filename}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            # --- UPDATED: Get all 3 return values ---
            label, confidence, predicted_index = model_predict(temp_path, MODEL)
            advice = TREATMENT_ADVICE.get(label, "Consult a local agricultural expert for specific treatment plans.")
            
            image_url = url_for('static', filename='uploads/' + filename)
            
            # --- NEW: Generate Heatmap ---
            heatmap_url = None
            if predicted_index != -1: # Only generate heatmap if prediction is confident
                heatmap_url = get_gradcam_heatmap(temp_path, predicted_index)
            # --- End of New Code ---

            prediction_data = {
                'label': label,
                'confidence': float(confidence),
                'advice': advice,
                'image_path': image_url,
                'heatmap_path': heatmap_url # Pass the heatmap URL to the template
            }
            
            # Save to database
            try:
                db = get_db()
                db_image_path = 'uploads/' + filename
                db.execute(
                    'INSERT INTO predictions (image_path, label, confidence, advice) VALUES (?, ?, ?, ?)',
                    (db_image_path, label, float(confidence), advice)
                )
                db.commit()
            except Exception as e:
                print(f"Database Error: {e}")
            
    return render_template('index.html', prediction=prediction_data)

# --- NEW: History Page Route ---
@app.route('/history')
def history():
    """Fetches all predictions from the database and displays them on the history page."""
    try:
        db = get_db()
        cursor = db.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
        predictions = cursor.fetchall()
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        print(f"History Page Error: {e}")
        return "Error loading history."
# --- End of New Code ---


if __name__ == '__main__':
    # UPDATED: Create the correct folder
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # NEW: Initialize the database on first run
    init_db() 
    
    app.run(debug=True)