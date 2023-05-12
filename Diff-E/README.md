# Diff-E: Diffusion-based Learning for Decoding Imagined Speech EEG
Decoding EEG signals for imagined speech is a challenging task due to the high-dimensional nature of the data and low signal-to-noise ratio. In recent years, denoising diffusion probabilistic models (DDPMs) have emerged as promising approaches for representation learning in various domains. Our study proposes a novel method for decoding EEG signals for imagined speech using DDPMs and a conditional autoencoder named Diff-E. Results indicate that Diff-E significantly improves the accuracy of decoding EEG signals for imagined speech compared to traditional machine learning techniques and baseline models. Our findings suggest that DDPMs can be an effective tool for EEG signal decoding, with potential implications for the development of brain-computer interfaces that enable communication through imagined speech.

This work is submitted to [Interspeech 2023](https://www.interspeech2023.org/). It is currently under review.
## EEG Classification with DDPM and Diff-E
The code implementation is based on repositories [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST).

This repository provides an implementation of an EEG classification model using Denoising Diffusion Probabilistic Model (DDPM) and Diffusion-based Encoder (Diff-E). The model is designed for 13-class classification of EEG signals for imagined speech.

### Main Function Description
The main function of this implementation (train) is responsible for training and evaluating the EEG classification model. The implementation is divided into the following steps:

1. Loading and Preparing Data: The data is loaded using the `load_data`, and split into training and testing sets using the `get_dataloader`. The batch size and path to the data should be specified.

2. Defining the Model: The model consists of four main components: DDPM, Encoder, Decoder, and Linear Classifier. Their dimensions and parameters should be specified before training.

3. Loss Functions and Optimizers: The implementation uses L1 Loss for training the DDPM and Mean Squared Error Loss for the classification task. `RMSprop` is used as the optimizer for both DDPM and Diff-E, and `CyclicLR` is employed as the learning rate scheduler.

4. Exponential Moving Average (EMA): EMA is applied to the Linear Classifier to improve its generalization during training.

5. Training and Evaluation: The model is trained for a specified number of epochs. During training, DDPM and Diff-E are optimized separately, and their loss functions are combined using a weighting factor (α (alpha)). The model is evaluated on the test set at regular intervals, and the best performance metrics are recorded.

6. Command Line Arguments: The main function accepts command-line arguments for specifying the number of subjects to process and the device to use for training (e.g., `'cuda:0'`).

## Using a Conda Environment

We encourage you to use a conda environment to manage your dependencies and create an isolated workspace for this project. This will help you avoid potential conflicts with other packages installed on your system.

### Installing Conda

If you don't have conda installed, you can download [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Follow the installation instructions for your platform.

### Setting Up a Conda Environment

Once you have conda installed, you can create a new environment and install the required packages by following these steps:

1. Clone the repository:
```bash
$ git clone https://github.com/diffe2023/Diff-E.git
$ cd yourrepository
```

2. Create a new conda environment:

```bash
$ conda create --name your_environment_name python=3.8
```
Replace `your_environment_name` with a name of your choice.

3. Activate the new environment:
- On Windows:
  ```
  $ conda activate your_environment_name
  ```
- On macOS and Linux:
  ```bash
  $ source activate your_environment_name
  ```
4. Install the required packages:

The following Python packages are required to run this project:

- `einops`
- `ema_pytorch`
- `mat73`
- `numpy`
- `scikit_learn`
- `torch`
- `tqdm`

5. Now you can run the `main.py` script within the conda environment:

```bash
$ python main.py --num_subjects <number_of_subjects> --device <device_to_use>
```
Replace `<number_of_subjects>` with the number of subjects you wish to process and `<device_to_use>` with the device you want to use for training, such as `'cuda:0'` for the first available GPU.

When you're done working with the conda environment, you can deactivate it with the following command:

```bash
$ conda deactivate
```

This will return you to your system's default environment.

## Pre-trained Model Evaluation
### Download
The pre-trained model and testset for subject 2 is provided [here (model)](https://anonymfile.com/7PzVe/diffe-2.pt) and [here (testset)](https://anonymfile.com/XPzxj/data-loader.pkl) for download. 

### Evaluation
Here's an example command to run the script:

```bash
python evaluation.py --model_path model.pt --data_loader_path data_loader.pkl
```

## Todo
- [x] Item 1: Streamline the code
- [ ] Item 2: Document the code
- [x] Item 3: Provide pre-trained models
- [ ] Item 4: Test on public datasets
- [ ] Item 5: Experiment on adding temporal convolutional layers
