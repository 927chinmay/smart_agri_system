import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model 
import os

DATA_DIR = r'./plantvillage dataset/color' 

IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32

MODEL_SAVE_PATH = 'smart_agri_weights.weights.h5' 

print("Starting Data Preparation...")

# 1. Image Data Generator (Normalization and Augmentation
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2, 
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

# (38)
NUM_CLASSES = train_generator.num_classes
print(f"Number of Disease Classes Detected: {NUM_CLASSES}")


# --- Model Development (MobileNetV2)

# 1. Load the Model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, 
    input_shape=(224, 224, 3) 
)
base_model.trainable = False # Freeze the base layers

# 2. Define the Model using the Functional API
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x) 
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

#  final model
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compiling
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model Summary (MobileNetV2) ---")
model.summary()


#  Model Training (On Local CPU) 
print(f"\nStarting Model Training on CPU (saving weights to: {MODEL_SAVE_PATH})...")
print("This model is highly efficient, but expect each epoch to take significant time on a CPU.")

# Callbacks

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, 
    monitor='val_accuracy', 
    save_best_only=True,
    save_weights_only=True, 
    verbose=1
)

# Train We run for 20 epochs maximum.
history = model.fit(
    train_generator,
    epochs=20, 
    validation_data=validation_generator,
    callbacks=[model_checkpoint]
)

print(f"\nModel Training Complete! The best weights are saved as '{MODEL_SAVE_PATH}'")