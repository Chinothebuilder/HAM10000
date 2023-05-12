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

Several adjustments were made to further optimize the model. These changes includes:
To learn new features from the data set, the xception model layers were left unfrozen during the training process in Google Colab.

Swapping out the top layers of the xception model with a Relu activated block consisting of a dropout layer (0.5), a fully connected dense layer (128) with a batch normalization layer. 

In order to balance out the class disparity, a class-weighted learning technique was adopted, which involved assigning different weights to different classes in the loss function. The weights were calculated using the class_weight function from the sklearn.utils libraries. 

To avoid over fitting call_back functions like the EarlyStopping, ReduceLRONPlateau and ModelCheckpoint were used during the training process.
Focal loss function was applied instead of the categorical cross_entropy to improve the model performance even further.

The problem of class imbalance was naturally resolved by focal loss. In real-world applications, we employ a α- balanced variant of the focal loss Ɣ that combines the traits of the focusing and weighing parameters, producing accuracy that is marginally higher than the non-balanced crossentropy loss

To further enhance the performance of the model, a soft-attention network layer with a dropout (0.2) was added to the architecture, this helps to enhance the value of critical features while suppressing the noise-inducing features. 


With these ajustments we were able to create eight alternative methods: 
CNN alone, 

CNN + Dropout regularization, 

CNN + Dropout regularization + Augmentation, 

CNN + Dropout + Augmentation + class_weights, 

CNN + Dropout + Augmentation + class_weights + soft attention layer, 
Xception + Dropout + Augmentation + class_weights, 
Xception + Dropout + Augmentation + class_weights + soft attention, 
Xception + Dropout + Augmentation + class_weights + soft attention + focal loss


