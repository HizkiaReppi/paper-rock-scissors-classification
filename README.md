# Kelas Belajar Machine Learning Untuk Pemula - Dicoding

Kelas Belajar Machine Learning Untuk Pemula | Alur Belajar Machine Learning Developer | Dicoding

Kelas ini membahas mengenai dasar dan algoritma-algoritma Machine Learning, serta mampu mengimplementasikannya dalam membuat model Machine Learning untuk memproses data.

Materi yang dipelajari dikelas ini:

- **Pengenalan Data** : Pengenalan ke machine learning dan teknik-teknik untuk pengolahan data, seperti data collecting, data cleaning, dan data processing.
- **Supervised dan Unsupervised Learning** : Memahami 2 jenis machine learning yaitu supervised dan unsupervised learning, dengan contoh model regresi linear dan decision tree.
- **Support Vector Machine (SVM)** : Menjelaskan tentang SVM, salah satu model machine learning yang populer. Di sini juga akan belajar tentang clustering dengan k-means.
- **Dasar-Dasar Machine Learning** : Menjelaskan tentang alur kerja (workflow) dari suatu proyek machine learning, dan juga menjelaskan overfitting, underfitting, dan model selection.
- **Neural Network** : Belajar mengenal dasar dari neural network. Akan diterangkan mengenai multi layer perceptron serta convolutional neural network dalam image classification.
- **TensorFlow** : Belajar tentang library TensorFlow, sebuah powerful library yang dipakai untuk mengembangkan project machine learning.

## Submission

Pada submission kelas ini, saya membuat model Machine Learning untuk proses klasifikasi gambar Paper Rock and Scissors (Gunting Batu Kertas) dengan menggunakan Convolutional Neural Network (CNN).

Create with ❤ | Hizkia Reppi

# Paper Rock Scissors Image Classification - Dicoding

## Overview

This project focuses on developing an image classification model for classifying hand gestures into three categories: Paper, Rock, and Scissors. The goal is to create an accurate and robust model capable of distinguishing between these gestures. The project utilizes a deep learning approach to achieve high accuracy.

## Dataset

The dataset used for this project consists of images depicting various hand gestures representing Paper, Rock, and Scissors. The dataset is curated and labeled to facilitate supervised learning. It is crucial to have a diverse and well-labeled dataset to train a model effectively.

Dataset Source: [Paper Rock Scissors Dataset](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip) from [Dicoding](https://dicoding.com)

## Model and Training

The image classification model is built using a convolutional neural network (CNN) architecture. CNNs are suitable for image-related tasks and have proven effective in learning hierarchical features. The model is trained on the labeled dataset using an appropriate training-validation split.

Training Parameters:
- Number of Epochs: 40
- Optimizer: Adam

The training process involves feeding the images through the network, adjusting the model's weights using backpropagation, and optimizing for accurate predictions.

## Evaluation

The model's performance is evaluated on a separate test set not seen during training. The evaluation metrics include accuracy, precision, recall, and F1 score. The goal is to achieve an accuracy of over 95%.

## Conclusion

This project successfully demonstrates the application of deep learning techniques for image classification, specifically in the context of Paper, Rock, and Scissors hand gestures. The achieved accuracy of over 95% indicates the model's effectiveness and its potential for use in various applications, such as gesture-based interfaces and games. Further optimizations and enhancements can be explored to improve the model's performance and capabilities.
