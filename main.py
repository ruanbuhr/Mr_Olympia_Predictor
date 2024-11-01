from flask import Flask, request, jsonify
import os
from model import *

app = Flask(__name__)

predictor = BodybuilderPredictor("model/model.h5")

@app.route('/rank', methods=['POST'])
def rank_bodybuilders():
    files = request.files.getlist("images")

    if len(files) < 2:
        return jsonify({"error": "Atleast two images are required"}), 400
    
    images = []
    for file in files:
        image = tf.image.decode_image(file.read(), channels=3)
        preprocessed_image = predictor.preprocess_image(image)
        images.append(preprocessed_image)

    ranking_indices = predictor.rank_bodybuilders(images)

    ranked_files = [files[i].filename for i in ranking_indices]
    return jsonify({"ranking": ranked_files})

if __name__ == '__main__':
    app.run(port=5000)