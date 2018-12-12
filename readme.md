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
- python3
- tensorflow
- keras
- imageio
- argparse
- requests
- tarfile
- numpy
- scipy
- pandas

### Downloading and processing IMDB/WIKI data
To download the IMDB/WIKI dataset:
```
python data-download.py
```

To process the IMDB/WIKI dataset for eventual ingestion by Keras ImageDataGenerator:
```
python data-process.py
```

You can specify train/validation/test splits through the following flags, though the defaults should be sufficient. The test proportion is inferred.
```
python data-process.py --train_prop 0.8 --validation_prop 0.1
```

For additional information:
```
python data-process.py --help
```

### Downloading and processing CELEB-A data

```
sh download_celebA.sh
```

### Training an SVAE on IMDB/WIKI data
All saved models are co-located with relevant Tensorboard logs, hyperparameter logs, and evaluation results in a unique experiment directory given at time of training. If the experiment directory already exists and contains a checkpoint, then that checkpoint is loaded to extend training rather than create a new model.

To train the encoder and classifier networks with default hyperparameters:
```
python train-svae.py --experiment_dir ../experiments/my-first-experiment/
```

To train the decoder network with default hyperparameters:
```
python train-svae.py --experiment_dir ../experiments/my-first-experiment/ --decoder
```

To evaluate a trained model on the test set:
```
python train-svae.py --experiment_dir ../experiments/my-first-experiment/ --test
```

For additional information:
```
python train-svae.py --help
```
