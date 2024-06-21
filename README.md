# CIFAR-10 Image Classification with CNN 

This repository implements a Convolutional Neural Network (CNN) model for classifying images from the CIFAR-10 dataset. The model achieves an accuracy of 71.37% on the test set.

The CIFAR-10 dataset is a popular benchmark for image classification, containing 60,000 32x32 color images in 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

## Dependencies:
TensorFlow , Keras , matplotlib , NumPy , OpenCV 

##  Code Structure:

* CNN_classifier_.ipynb: Contains the core code for model definition, training, evaluation, and prediction.
* Test_images: Contains 2 random images from internet I finally tested the model on 

## Model Architecture:

This model uses a sequential architecture with convolutional and dense layers:

* Input Layer: Takes a 32x32x3 RGB image as input.
* Convolutional Layers:
    * Two convolutional layers with 32 and 64 filters, respectively, each using a 3x3 kernel and ReLU activation.
    * Max pooling layers (2x2) are applied after each convolutional layer for downsampling.
* Flatten Layer: Flattens the output of the convolutional layers into a 1D vector.
* Dense Layers:
   * A dense layer with 64 units and ReLU activation for feature extraction.
   * A final dense layer with 10 units and softmax activation for predicting class probabilities.

## Training Process:

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metrics: Accuracy
* Epochs: 15
* Validation Split: A portion of the training data is used for validation to monitor performance during training.

## Evaluation:

The model achieves an accuracy of 71.37% on the test set. You can also visualize the training and validation loss/accuracy curves over epochs using matplotlib for further analysis.

## Conclusion:

This repository demonstrates a basic CNN approach to CIFAR-10 image classification. You can explore hyperparameter tuning, data augmentation, and more complex architectures for potentially higher accuracy.


