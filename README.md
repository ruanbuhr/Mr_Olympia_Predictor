# Mr Olympia Predictor

This application leverages machine learning to rank bodybuilders on their likelihood of winning the Mr Olympia competition.

## Overview

The Mr. Olympia Predictor is powered by a Convolutional Neural Network (CNN) that has been trained on bodybuilding images to recognize key physique attributes. Given images of bodybuilders in similar poses, the model determines which physique is more likely to be favored in a competitive setting.

## Machine Learning

### Model Objective

The model’s primary goal is to differentiate between physiques based on popular bodybuilding poses and identify the physique most likely to win.

### Dataset

The dataset consists of manually gathered photos from the 2024 Mr. Olympia competition. Each record in the dataset compares two images of bodybuilders performing the same pose, and a label indicates which bodybuilder won that particular comparison.

| Bodybuilder A   | Bodybuilder B    | Won |
| --------------- | ---------------- | --- |
| cbum-side-chest | ramon-side-chest | 1   |

In the example above, "Cbum" wins the side-chest pose against "Ramon Dino", so `Won` is marked as `1`. Had "Ramon" won, `Won` would have been marked as `0`.

### Training the Model

The model was trained on this dataset to learn the differences in physiques through various standard bodybuilding poses and favor the winning physique.

## Getting Started

### Prerequisites

- **Python** 3.12 or above
- **Tensorflow** for the CNN model
- **Flask** to serve the model via a REST API

### Training the Model

To train your own model, you need a dataset formatted as described. The current model requires manually labeled images of bodybuilders in identical poses with a binary win/loss indicator. Note that creating a custom dataset and training a model from scratch is essential for the app to work optimally.

Inside a data/ directory in the root of the repo you need to create a dataset.csv which contains the data and then an images/ folder which contains the images.

### Running the Application

The application is encoded into a Rest API which you can use by running:

```
python3 main.py
```

from within the root of the repo.

Once the server is running to can make post requests to the following endpoint:

```
POST https://localhost:5000/rank
```

In the request body, upload the images of bodybuilders you want ranked. The API will return the images in descending order of the model’s prediction for likelihood to win.

## Options for Improvement

### Data Collection

The primary challenge for the Mr. Olympia Predictor is the availability of high-quality, labeled data. Bodybuilding comparison datasets are rare, so manually curating and expanding the dataset with more pose variations and bodybuilders would significantly improve the model’s accuracy.
