import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 # <-- Fast, efficient model for CPU
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model 
import os

# --- Configuration ---
# !!! IMPORTANT: VERIFY THIS PATH ON YOUR LOCAL MACHINE !!!
# This path must point to the folder containing the 38 disease subfolders (e.g., Apple___Black_rot)
DATA_DIR = r'./plantvillage dataset/color' 

IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32
# This is the standard, robust save format for deployment (weights only)
MODEL_SAVE_PATH = 'smart_agri_weights.weights.h5' 

print("Starting Data Preparation...")

# 1. Image Data Generator (Normalization and Augmentation)
datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values from 0-255 to 0-1
    validation_split=0.2, # Hold out 20% for validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 2. Data Generators
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' 
)
validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get the number of classes (38)
NUM_CLASSES = train_generator.num_classes
print(f"Number of Disease Classes Detected: {NUM_CLASSES}")


# --- Model Development (MobileNetV2 Functional API) ---

# 1. Load the Pre-trained Base Model (MobileNetV2)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, # Exclude the original classification head
    input_shape=(224, 224, 3) 
)
base_model.trainable = False # Freeze the base layers

# 2. Define the Model using the Functional API (Robust structure)
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x) # Feature refinement
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model Summary (MobileNetV2) ---")
model.summary()


# --- Model Training (On Local CPU) ---
print(f"\nStarting Model Training on CPU (saving weights to: {MODEL_SAVE_PATH})...")
print("This model is highly efficient, but expect each epoch to take significant time on a CPU.")

# Callbacks
# ModelCheckpoint is used to save the best model weights found so far
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, 
    monitor='val_accuracy', 
    save_best_only=True,
    save_weights_only=True, # CRUCIAL: Saves only weights to avoid serialization error
    verbose=1
)

# Train the model! We run for 20 epochs maximum.
history = model.fit(
    train_generator,
    epochs=20, 
    validation_data=validation_generator,
    callbacks=[model_checkpoint]
)

print(f"\nModel Training Complete! The best weights are saved as '{MODEL_SAVE_PATH}'")