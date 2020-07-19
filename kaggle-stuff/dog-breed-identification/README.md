# Dog-breed Identification

* Solutions to [the "dog-breed identification" problem on Kaggle](https://www.kaggle.com/c/dog-breed-identification), where various dog images are categorized into one of 120 categories. 

* Examples demonstrating the difference between simple CNN and various pretrained deep networks, implemented with [Keras](https://keras.io/).


## Prerequisites:
- Python 3
- Keras 2.2.2
- Tensorflow 1.10.1
- GPU

## Run

- Download the data from [https://www.kaggle.com/c/dog-breed-identification/data](https://www.kaggle.com/c/dog-breed-identification/data)

- Upzip the data into the `./data` where `./data/train` contains all the training images, `./data/test` contains all the testing images, and `./data/labels.csv` contains the training labels

- Run `./data_preprocessing.ipynb`, this groups all the training images according into their respective categories in the `./data_gen` directory


#### Simple CNN:
- Run `./simple_conv.ipynb` to run a simple convolutional neural network on the training data and see the results.

#### Pretrained CNNs:
- Run `./feature_extraction.ipynb` to extract bottleneck features of the training images, along with their labels, and store them in the `./features` directory. Select one of the many existing CNN model that come with Keras, and extract the corresponding features.

- Run `./pretrained.ipynb` to train and validate on the features.


## Remarks

* Keras' built-in ImageDataGenerator is used to generate training and validation data for both the simple CNN and feature extraction. Note that during feature extraction, do not rescale the image pixel value with *1./255* as pretrained models' *preprocess_input* takes care of that.

* Regularization, drop-out and batch normalization are utilized to counter overfitting.



* As expected, a simple CNN is inadequate to categorize 120 dog breeds on a relatively small dataset (roughly 10,000 examples). Both the training accuracy and validation accuracy fall around 0.01, random basically.

* Using features extracted from pretrained models as input of a simple dense network is already effective. It is interesting to see different pretrained models perform differently. (All accuracy scores are obtained by running Adam optimizer with LR=1e-3 for 60 epochs, then with LR=1e-5 for 60 epochs.)

	| Models        | Train Acc.    | Validation Acc.  |
	| ------------- |:-------------:| -----:|
	| VGG19         | 97%           |   63% |
	| ResNet50      | 99%           |   69% |
	| Xception      | 99%       	  |   81% |
	| InceptionResNetV2 | 99%       |   86% | 






 




