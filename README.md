# Hand Sign Recognition with Machine Learning

## Overview
This project aims to recognize hand signs using machine learning techniques. The dataset consists of images capturing hand signs for letters A, B, C, as well as common gestures like "Hi," "Love you," "Thank you," "Yes," and "No."

## Dataset
The dataset used in this project can be found [here](https://drive.google.com/drive/folders/1KPJ0pEeOGA4Xc8aMkmlJ-DoQti09pE6K?usp=sharing).

## Dataset Creation
The dataset was created by capturing photos of hand signs using a camera. Each class (letter or gesture) was represented by a set of images taken in various lighting conditions and angles.

## Model Training
The dataset was split into training and testing sets. Three different machine learning models were trained using the training data:
- Random Forest Classifier
- K-Nearest Neighbors (KNN) Classifier
- Support Vector Machine (SVM) Classifier

Each model was trained to recognize hand signs based on the extracted features from the images.

## Model Evaluation
After training the models, they were evaluated using the testing data to assess their performance and accuracy in recognizing hand signs. The accuracy of each model was calculated and compared to determine the most effective approach for hand sign recognition.

## Files
- `create_dataset.py`: Python script for capturing images and creating the dataset.
- `train_classifier.py`: Python script for training machine learning models.
- `data.pickle`: Pickle file containing the preprocessed dataset.
- `RF_model.p`: Pickle file containing the trained Random Forest model.
- `SVM_model.p`: Pickle file containing the trained SVM model.
- `knn_model.p`: Pickle file containing the trained KNN model.

## Usage
To replicate the project:
1. Run `create_dataset.py` to capture images and create the dataset.
2. Run `train_classifier.py` to train the machine learning models.
3. Test the trained models using the testing data to evaluate their performance.

## Dependencies
- OpenCV
- Mediapipe
- Scikit-learn
- NumPy
- Matplotlib (optional for visualization)
