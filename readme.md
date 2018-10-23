# Supervised Variational Autoencoders

### Project Organization

------------
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── svae               <- Supervised VAE package
    │   └── __init__.py    
    │
    ├── data
    │   ├── temp           <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── experiments        <- Trained and saved models, evaluation results, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a two digit number (for ordering),
    │                         the creator's initials, and a short `-` delimited description,
    │                         e.g. `01-dz-initial-data-exploration`.
    │
    ├── reports            <- Analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── scripts            <- Scripts for use in this project (e.g. data processing, training,
    │                         testing, visualizations, etc.)
    │
    ├── tests              <- Pytest testing suites
--------

### Package Management

Use conda to maintain a virtual environment.

### Style Guide

Follow [PEP 8][], when sensible.

#### Naming

- Variables, functions, methods, packages, modules
    - `lower_case_with_underscores`
- Classes and Exceptions
    - `CapWords`
- Protected methods and internal functions
    - `_single_leading_underscore(self, ...)`
- Private methods
    - `__double_leading_underscore(self, ...)`
- Constants
    - `ALL_CAPS_WITH_UNDERSCORES`

#### Line lengths

80 characters maximum per line.

Use parentheses for line continuations.

```python
wiki = (
    "The Colt Python is a .357 Magnum caliber revolver formerly manufactured "
    "by Colt's Manufacturing Company of Hartford, Connecticut. It is sometimes "
    'referred to as a "Combat Magnum". It was first introduced in 1955, the '
    "same year as Smith & Wesson's M29 .44 Magnum."
)
```

#### Indentation

Use **two** spaces instead of four as Tensorflow code can be very long.

#### Documentation

Use numpy-style documentation. 
