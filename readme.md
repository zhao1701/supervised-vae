# Supervised Variational Autoencoders
In the video below for Columbia University's **COMS 4995: Applied Deep Learning** course, we present some background for the use of variational autoencoders (VAEs) in gaining neural network interpretability and experiment with a new architecture to apply VAE concepts to a supervised setting.

[![Project overview video](https://img.youtube.com/vi/x294DLH3-Cs/0.jpg)](https://www.youtube.com/watch?v=x294DLH3-Cs)

While the quality of latent traversals was not what we hoped for, we found surprising and preliminary results showing that the variational layer may behave as an effective regularizer.

- **Contributors**: Ramy Fahim, Serena Zhang, Derek Zhao
- **Mentors**: Josh Gordon (Google), Loc Tran (NASA Langley Research Center)
- **Locations**: Viacom

### Project Organization

------------
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── svae               <- Supervised VAE package
    │   └── __init__.py    <- Contains the SVAE class
    │
    ├── svae_mnist         <- Supervised VAE package with architecture modified for MNIST-sized data
    │   ├── __init__.py
    │   └── testpad.py     <- This wound up serving as the script for training the SVAE on MNIST data
    │
    ├── data     
    │   ├── CELEBA         <- Processed image data from celebA dataset
    │   │   ├── sample/
    │   │   ├── train/
    │   │   ├── test/
    │   │   └── validation/
    │   │
    │   ├── temp           <- Intermediate data that has been transformed
    │   ├── processed      <- Processed image data from IMDB/WIKI dataset
    │   │   ├── sample/
    │   │   ├── train/
    │   │   ├── test/
    │   │   └── validation/
    │   └── raw            <- The original, immutable data dump.
    │
    ├── experiments        <- Trained and saved models, evaluation results, and model summaries.
    │   └── experiment-01  <- Experiment directory containing a Tensorflow checkpoint
    │       ├── checkpoints/    <- Saved models
    │       ├── logs/           <- Tensorboard logs
    │       └── traversals/     <- Animated gifs of latent traversals
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a two digit number (for ordering),
    │                         the creator's initials, and a short `-` delimited description,
    │                         e.g. `01-dz-initial-data-exploration`.
    │
    ├── scripts            <- Scripts for use in this project (e.g. data processing, training,
    │                         testing, visualizations, etc.)
    │
    └── tests              <- Pytest testing suites
--------

### Dependencies

- tensorflow
- keras
- imageio
- argparse
- requests
- tarfile
- numpy
- scipy
- pandas
