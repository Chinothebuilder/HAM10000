# HAM10000

Introduction:

This project aims to compare the performance of popular deep learning models, Convolutional Neural Network (CNN) and Xception with their added architectural modifications, for image classification on the Ham10000 dataset. The Ham10000 dataset contains 10,015 dermatoscopic images of pigmented skin lesions, which are categorized into seven different types of skin cancer.

Setup:

The project code is available on GitHub and is implemented in Python using TensorFlow and Keras libraries. The dataset is preprocessed and split into train, validation, and test sets using a 80-10-10 split ratio. The CNN and Xception models are trained on the training set, validated on the validation set, and evaluated on the test set.

Models:

The CNN model consists of four convolutional layers with max-pooling, followed by two fully connected layers with BatchNormalization and dropout regularization. The Xception model is a pre-trained deep learning model that has been trained on ImageNet and includes 126 layers. Transfer learning using Xception network was employed as a feature extractor for the categorization of skin lesions.  The rational to this is that xception which in the family of inception networks from google labs has following merit. 

Reduced model size and computational cost 

Gain in performance on CNN than other inception models
Trains more quickly than the VGG family
Accepts a lesser image dimension of (71, 71) 
