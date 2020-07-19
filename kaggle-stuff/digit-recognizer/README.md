# Digit Recognizer

* Solutions to the good old "hand-written digit recognition" problem, using MNIST data, as a [Kaggle competition](https://www.kaggle.com/c/digit-recognizer)

* Toy examples demonstrating the difference and effectiveness of simple dense neural networks and convolutional neural networks, implemented with [Keras](https://keras.io/)

## Prerequisites:
- Python 3
- Keras 2.2.2
- Tensorflow 1.10.1
- GPU

----------
### Notes:

- For image classification problem, convolution nets are an obviously better choice than dense nets. But dense nets prove to be not far off. With the included implementations, the former achieves 99.4% accuracy (on 60,000 training examples) and the latter achieves 97.8% accuracy (on 42,000 training examples).

- Regularization is vital in tackling overfitting. Specifically L2 regularization is used for the dense net and drop-out is used for the conv net. 

- Both networks are 'shallow' nets comparing to modern ones, 4 layers for the dense net and 6 layers for the conv net. 

- Both networks are remarkably easy to implement with Keras framework.

- To cheat the competition, just train on all the available MNIST examples until it overfits, namely reaching 100% accuracy.