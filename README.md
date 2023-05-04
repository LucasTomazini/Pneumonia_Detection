
# Chest X-Ray Pneumonia Detection

This project is a pneumonia detection algorithm that uses chest X-ray images to predict if a patient has pneumonia. The algorithm uses PyTorch to create a custom dataset and train a LENET convolutional neural network (CNN) model. It also incorporates transfer learning techniques to train models based on the ResNet and VGG architectures.

## Dataset

The dataset used in this project is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset, which contains a total of 5,856 chest X-ray images from pediatric and adult patients. The images are labeled as either "normal" or "pneumonia".

## Getting Started

To use this algorithm, you will need to have PyTorch and the dependencies listed in the requirements.txt file installed on your computer. You will also need to download the Chest X-Ray Images (Pneumonia) dataset and save it to the same directory as the algorithm files.

## Usage

1. Open the `train.py` file in a Python IDE or run it from the command line.
2. The `train.py` file will load the data, preprocess it, and train a LENET CNN model on the training data.
3. Once the model is trained, it will be used to predict whether or not a patient has pneumonia on the test dataset.
4. To train a model using transfer learning with the ResNet or VGG architectures, open the corresponding `train_resnet.py` or `train_vgg.py` file and follow the same steps as above.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

This algorithm was developed as a project for pneumonia detection using chest X-ray images. It was based on the work of other researchers in the field of medical image analysis, and incorporates custom dataset creation, CNN training with PyTorch, and transfer learning techniques.
