# Hello NN

My Neural Network Codes when I was beginner in Artificial Neural Networks

Clean, Commented, Single-Filed Python Programs + Jupyter Notebooks

## Prerequisites

The programs depends upon following dependencies:

* Python 3.5+
   * Numpy
   * Matplotlib
   * Tensorflow

## Index


* FeedForward Neural Network
    * Classification `Iris dataset`
        * numpy - [code](Feedforward-NeuralNet/code/np_nn_iris_classification.py), [notebook](Feedforward-NeuralNet/np_nn_iris_classification.ipynb)
        * tensorflow - [code](Feedforward-NeuralNet/code/tf_nn_iris_classification.py), [notebook](Feedforward-NeuralNet/tf_nn_iris_classification.ipynb)

* Recurrent Neural Network
    * Addition `8-bit numbers`
        * numpy - [code](Recurrent-NeuralNet/code/np_rnn_addition.py), [notebook](Recurrent-NeuralNet/np_rnn_addition.ipynb)
        * tensorflow - [code](Recurrent-NeuralNet/code/tf_rnn_addition.py), [notebook](Recurrent-NeuralNet/tf_rnn_addition.ipynb)
    * Time-series `CO2(ppm) mauna loa, 1965-1980`
        * tensorflow - [code](Recurrent-NeuralNet/code/tf_lstm_climate_timeseries.py), [notebook](Recurrent-NeuralNet/tf_lstm_climate_timeseries.ipynb)

* Convolutional Neural Network
    * Classification `MNIST`
        * tensorflow - [code](Convolutional-NeuralNet/code/tf_cnn_mnist_classification.py), [notebook](Convolutional-NeuralNet/tf_cnn_mnist_classification.ipyn)

* Hopfield Network
    * Data Reconstruction
        * numpy - [code](Hopfield-Network/code/np_hnn_reconstruction.py), [notebook](Hopfield-Network/np_hnn_reconstruction.ipynb)

* Restricted Boltzmann Machine
    * Image Reconstruction `MNIST`
        * numpy - [code](Restricted-Boltzmann-Machine/code/np_rbm_mnist_reconstruction.py), [notebook](Restricted-Boltzmann-Machine/np_rbm_mnist_reconstruction.ipynb)
        * tensorflow - [code](Restricted-Boltzmann-Machine/code/tf_rbm_mnist_reconstruction.py), [notebook](Restricted-Boltzmann-Machine/tf_rbm_mnist_reconstruction.ipynb)

* Denoising Neural AutoEncoder
    * Image Denoising `MNIST`
        * tensorflow - [code](Denoising-Autoencoder/code/tf_dae_mnist_reconstruction.py), [notebook](Denoising-Autoencoder/tf_dae_mnist_reconstruction.ipynb)

* Deconvolutional Neural AutoEncoder
    * Image Reconstruction `MNIST`
        * tensorflow - [code](Deconvolutional-Autoencoder/code/tf_dcae_mnist_reconstruction.py), [notebook](Deconvolutional-Autoencoder/tf_dcae_mnist_reconstruction.ipynb)
