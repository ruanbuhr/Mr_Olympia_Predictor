import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    return image

def prepare_data(csv_path, images_folder):
    df = pd.read_csv(csv_path)
    
    # Initialize lists for storing image pairs and labels
    image_a_list, image_b_list, labels = [], [], []
    
    for _, row in df.iterrows():
        bodybuilder_a = row['BodybuilderA']
        bodybuilder_b = row['BodybuilderB']
        label = row['Won']
        
        image_a_path = Path(images_folder) / f"{bodybuilder_a}.png"
        image_b_path = Path(images_folder) / f"{bodybuilder_b}.png"
        
        image_a = load_and_preprocess_image(str(image_a_path))
        image_b = load_and_preprocess_image(str(image_b_path))
        
        image_a_list.append(image_a)
        image_b_list.append(image_b)
        labels.append(label)
    
    return image_a_list, image_b_list, labels

def create_base_network(input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    return model

def create_model(input_shape):
    base_network = create_base_network(input_shape)
    
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    merged = layers.concatenate([processed_a, processed_b])
    output = layers.Dense(1, activation='sigmoid')(merged)
    
    model = Model([input_a, input_b], output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(csv_path, images_folder, input_shape=(224, 224, 3), epochs=10, batch_size=32):
    image_a_list, image_b_list, labels = prepare_data(csv_path, images_folder)

    images_a_train, images_a_test, images_b_train, images_b_test, labels_train, labels_test = train_test_split(
        image_a_list, image_b_list, labels, test_size=0.2, random_state=42
    )
    
    # Convert training and testing sets to tensors
    images_a_train = tf.stack(images_a_train)
    images_b_train = tf.stack(images_b_train)
    labels_train = tf.convert_to_tensor(labels_train, dtype=tf.float32)
    
    images_a_test = tf.stack(images_a_test)
    images_b_test = tf.stack(images_b_test)
    labels_test = tf.convert_to_tensor(labels_test, dtype=tf.float32)
    
    model = create_model(input_shape)
    
    model.fit(
        [images_a_train, images_b_train], labels_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([images_a_test, images_b_test], labels_test)
    )
    
    model.save("model.h5")
    print("Model trained and saved at model/model.h5")

if __name__ == "__main__":
    train_model("../data/dataset.csv", "../data/images")
