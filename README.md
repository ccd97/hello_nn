# Hello NN

My Neural Network Codes when I was beginner in Artificial Neural Networks

Clean, Commented, Single-Filed Python Programs + Jupyter Notebooks

## Prerequisites

The programs require Python and the following packages:

| Package | Version | Used by |
| :--- | :--- | :--- |
| Python | 3.9+ | All programs |
| NumPy | 1.14+ | All implementations |
| Matplotlib | 2.0+ | Plotting outputs and graphs |
| TensorFlow | 2.0+ | `tf_*` implementations |
| PyTorch | 2.0+ | `torch_*` implementations |
| torchvision | 0.15+ | MNIST loading in the PyTorch programs |
| Jupyter Notebook | — | Running the `.ipynb` notebooks |

Install the Python packages with:

```bash
pip install numpy matplotlib tensorflow torch torchvision jupyter
```

TensorFlow and PyTorch are only needed for the corresponding implementations, so you can omit either framework if you don't plan to run those programs.

## Index

| Neural Network | Task | NumPy | TensorFlow | PyTorch |
| :--- | :--- | :--- | :--- | :--- |
| FeedForward Neural Network | Classification `Iris dataset` | [code](Feedforward-NeuralNet/code/np_nn_iris_classification.py) · [notebook](Feedforward-NeuralNet/np_nn_iris_classification.ipynb) | [code](Feedforward-NeuralNet/code/tf_nn_iris_classification.py) · [notebook](Feedforward-NeuralNet/tf_nn_iris_classification.ipynb) | [code](Feedforward-NeuralNet/code/torch_nn_iris_classification.py) · [notebook](Feedforward-NeuralNet/torch_nn_iris_classification.ipynb) |
| Recurrent Neural Network | Addition `8-bit numbers` | [code](Recurrent-NeuralNet/code/np_rnn_addition.py) · [notebook](Recurrent-NeuralNet/np_rnn_addition.ipynb) | [code](Recurrent-NeuralNet/code/tf_rnn_addition.py) · [notebook](Recurrent-NeuralNet/tf_rnn_addition.ipynb) | [code](Recurrent-NeuralNet/code/torch_rnn_addition.py) · [notebook](Recurrent-NeuralNet/torch_rnn_addition.ipynb) |
| Recurrent Neural Network | Time-series `CO2(ppm) mauna loa` | — | [code](Recurrent-NeuralNet/code/tf_lstm_climate_timeseries.py) · [notebook](Recurrent-NeuralNet/tf_lstm_climate_timeseries.ipynb) | [code](Recurrent-NeuralNet/code/torch_lstm_climate_timeseries.py) · [notebook](Recurrent-NeuralNet/torch_lstm_climate_timeseries.ipynb) |
| Convolutional Neural Network | Classification `MNIST` | — | [code](Convolutional-NeuralNet/code/tf_cnn_mnist_classification.py) · [notebook](Convolutional-NeuralNet/tf_cnn_mnist_classification.ipynb) | [code](Convolutional-NeuralNet/code/torch_cnn_mnist_classification.py) · [notebook](Convolutional-NeuralNet/torch_cnn_mnist_classification.ipynb) |
| Hopfield Network | Data Reconstruction | [code](Hopfield-Network/code/np_hnn_reconstruction.py) · [notebook](Hopfield-Network/np_hnn_reconstruction.ipynb) | — | — |
| Restricted Boltzmann Machine | Image Reconstruction `MNIST` | — | [code](Restricted-Boltzmann-Machine/code/tf_rbm_mnist_reconstruction.py) · [notebook](Restricted-Boltzmann-Machine/tf_rbm_mnist_reconstruction.ipynb) | — |
| Denoising Neural AutoEncoder | Image Denoising `MNIST` | — | [code](Denoising-Autoencoder/code/tf_dae_mnist_reconstruction.py) · [notebook](Denoising-Autoencoder/tf_dae_mnist_reconstruction.ipynb) | [code](Denoising-Autoencoder/code/torch_dae_mnist_reconstruction.py) · [notebook](Denoising-Autoencoder/torch_dae_mnist_reconstruction.ipynb) |
| Deconvolutional Neural AutoEncoder | Image Reconstruction `MNIST` | — | [code](Deconvolutional-Autoencoder/code/tf_dcae_mnist_reconstruction.py) · [notebook](Deconvolutional-Autoencoder/tf_dcae_mnist_reconstruction.ipynb) | [code](Deconvolutional-Autoencoder/code/torch_dcae_mnist_reconstruction.py) · [notebook](Deconvolutional-Autoencoder/torch_dcae_mnist_reconstruction.ipynb) |
