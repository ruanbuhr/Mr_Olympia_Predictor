import tensorflow as tf
import numpy as np
from pathlib import Path

class BodybuilderPredictor:
    def __init__(self, model_path="model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def preprocess_image(self, image, target_size=(224, 224)):
        image = tf.image.resize(image, target_size)
        image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
        return np.expand_dims(image, axis=0)  # Add batch dimension
    
    def predict_winner(self, image_a, image_b):
        prediction = self.model.predict([image_a, image_b])
        return 1 if prediction >= 0.5 else 0  # Returns 1 if image_a wins, otherwise 0
    
    def rank_bodybuilders(self, images):
        n = len(images)
        win_counts = {i: 0 for i in range(n)}  # Initialize win count for each image

        # Compare each pair of images
        for i in range(n):
            for j in range(i + 1, n):
                winner = self.predict_winner(images[i], images[j])
                if winner == 1:
                    win_counts[i] += 1
                else:
                    win_counts[j] += 1

        # Sort images by win counts in descending order
        sorted_indices = sorted(win_counts.keys(), key=lambda x: win_counts[x], reverse=True)
        return sorted_indices
