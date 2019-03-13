# Lelantos #
A deep learning attempt at splitting phrases into clauses.

## How to run ##

### Prerequisites ###
To execute this code you'll need to either activate [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) or run it on a linux OS.

As an alternative, you can install [VMware Workstation Player](https://www.youtube.com/watch?v=zGzcDkGgVe0) and then [install Ubuntu as a virtual machine](https://youtu.be/CdiKs6Hu9O4).

Install the following packages:
1. `python3`
2. `python3-dev`
3. `python3-venv`
4. `git`

### Clone the repository ###

Use `git clone` command to clone this repository.

``` shell
git clone https://github.com/RePierre/lelantos.git
```

### Setup virtual environment ###

In terminal, navigate to the root directory of the repository and issue the following command to create a virtual environment:

``` shell
python3 -m venv .venv
```

Activate the virtual environment using the following command:

``` shell
source .venv/bin/activate
```

Once the environment is activated, install the required packages:

``` shell
cd ./src/
pip install -e .
```

### Train the model ###

Activate the virtual environment:

``` shell
source .venv/bin/activate
```

Navigate to `src` directory:

``` shell
cd ./src/
```

Run the model with with default parameters using the following command:

``` shell
python seq2seq.py --corpus-dir <path-to-corpus-directory>
```

To specify additional parameters for the training run `python seq2seq.py -h` to see the list of input parameters and what each parameter does.

### View model evolution in TensorBoard ###

1. Open a new terminal and activate the virtual environment.
2. Once activated, run `tensorboard --logdir <path-to-logs-folder>`. By default, `<path-to-logs-folder>` is `./src/logs/`.
3. Open a web browser and navigate to `http://localhost:6006`


## References ##

1. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
2. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
3. [Keras activation functions](https://keras.io/activations/)
4. [TensorBoard](https://github.com/tensorflow/tensorboard)

## Meaning of the name ##

Lelantos was a titan in Greek mythology. His name means **something that goes unobserved**.[^1]

The scope of this project is to find **something that goes unobserved** in a phrase, namely /the marker between two clauses/.

[^1]: https://www.greekmythology.com/Titans/Lelantos/lelantos.html
