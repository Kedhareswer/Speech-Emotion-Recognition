# Speech-Emotion-Recognition

## Abstract
The Speech Emotion Recognition (SER) project aims to identify human emotions through the pitch, tone, rhythm, and intensity of their speech. Emotions play a crucial role in communication, and the SER system brings similar emotional perception to human-computer interaction, customer service, health monitoring, and security applications. This project employs deep learning mechanisms, specifically Convolutional Neural Networks (CNN), to classify emotions in audio samples using datasets like RAVDESS, TESS, SAVEE, and CREMA-D. The model achieved a pre-training accuracy of 99.42%, showcasing its effectiveness in recognizing emotions from audio signals.

## Link : https://drive.google.com/drive/folders/1DCpOiD2FoD4ZDhXN4aE8d5d6y0W6nz_k

## Objective
The overall objective is to develop an efficient and reliable SER model using CNNs. The project focuses on accurate classification of emotions from audio samples and explores the potential applications in human-computer interaction, customer support, and mental health checks.

## Key Objectives:
Developing a Robust SER Model: Present an emotion recognition and classification method based on the CNN model.

Analyzing and Preprocessing Different Datasets: Combine and preprocess multiple publicly available datasets.

Apply Data Augmentation: Enhance model robustness with noise injection, pitch shifting, and speed alteration.

Feature Extraction and Audio Feature Engineering: Extract important audio features like ZCR, RMS, and MFCCs.

Enhance Model Generalization: Ensure the model generalizes well to new, unseen data.

Model Performance Testing: Measure accuracy, precision, recall, and F1-score to assess performance.

Practical Applications of SER: Research applications in various domains where emotional awareness is impactful.

Future Research in SER: Provide a foundation for future SER research with larger datasets and advanced models.

## Approach
Data Preparation
Dataset: Utilized RAVDESS, TESS, SAVEE, and CREMA-D datasets.

Preprocessing: Resized, normalized, and applied transformations to enhance model robustness.

## Model Architecture
Developed a deep neural network tailored to produce clear, detailed sketches.

Iteratively optimized the model to capture fine line work and shading.

## Training
Experimented with various neural network layers and configurations.

Fine-tuned hyperparameters to balance model complexity with training time.

## Evaluation
Used metrics such as Mean Absolute Error (MAE) and Structural Similarity Index (SSIM) to evaluate fidelity.

Conducted qualitative assessments to ensure accurate representation of key features and textures.

## Results
Effectively generated clear, hand-drawn-style sketches with high visual appeal.

Final sketches showcased intricate line details and shading, closely resembling traditional pencil sketches.

Demonstrated promising potential for applications in creative AI, portrait rendering, and more.

## Key Insights
The model successfully learned to generate consistent, hand-drawn-like sketches.

Fine-tuning layers and experimenting with hyperparameters were crucial to achieving high-quality sketches.

## Future Work
Experiment with larger datasets for broader generalization.

Extend the model to support different artistic styles beyond sketching.

## Getting Started
To get started with the SER project, follow these steps:

## Prerequisites
Python 3.x

TensorFlow

NumPy

librosa

Scikit-learn

## Usage
Data Preprocessing: Prepare the datasets and extract audio features.

Model Training: Train the CNN model with the preprocessed data.

Model Evaluation: Evaluate the model's performance using test datasets.

Prediction: Use the trained model to predict emotions from new audio samples.

## Contributing
Contributions are welcome! Please feel free to submit issues and pull requests to enhance the project.
