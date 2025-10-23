### Project Overview

This project implements a Convolutional Neural Network (CNN) model using Transfer Learning to classify chest X-ray images into two categories: NORMAL and PNEUMONIA. The model uses a pre-trained ResNet50 architecture fine-tuned for this specific medical imaging task.

##3Model Architecture

The classification task is achieved using a pre-trained ResNet50 model, which acts as a powerful feature extractor.

Base Model: ResNet50 (weights initialized from ImageNet).

Transfer Learning: The base layers of ResNet50 are frozen (trainable=False) to leverage its robust, pre-learned features for edge and texture detection, preventing overfitting on the smaller X-ray dataset.

Custom Head: A new classification head is added, consisting of:

GlobalAveragePooling2D

Dense layer (1024 units, 'relu' activation)

BatchNormalization

Dropout (0.5 rate)

Final Dense output layer (1 unit, 'sigmoid' activation) for binary classification.

###📊Dataset and Data Preparation

The model is trained on an X-ray image dataset structured into train, test, and val directories.

*Data Summary:*

The data generators found the following number of images across 2 classes (NORMAL, PNEUMONIA):

Training Images: 5216 images

Test Images: 624 images

Validation Images: 16 images

*Preprocessing and Augmentation:*

The ImageDataGenerator is used to load and prepare images for the ResNet50 model.

Normalization: All pixel values are rescaled to the range [0, 1] (rescale=1./255).

Target Size: Images are resized to (224, 224), which is the standard input size for ResNet50.

Data Augmentation (Training Only): To improve model robustness, the training data is heavily augmented using:

Horizontal/Vertical Flips

Rotation (rotation_range=25)

Zoom (zoom_range=0.3)

Shear (shear_range=0.2)

Brightness Adjustment (brightness_range=(0.8, 1.2))

###⚙️Technologies and Dependencies

The project is built using the following key libraries:

TensorFlow and Keras

numpy

pandas

matplotlib (for visualizations)

cv2 (OpenCV)

os


### 🚀 How to Run the Project

1.  **Clone the Repository**
    ```bash
    git clone [Your Repository URL]
    cd X-ray-Image-Classification
    ```

2.  **Install Dependencies**
    ```bash
    pip install tensorflow numpy pandas matplotlib opencv-python
    ```

