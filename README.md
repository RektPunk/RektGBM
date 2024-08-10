<p align="center">
  <img src="https://github.com/RektPunk/RektGBM/blob/main/assets/image/logo.png" alt="logo" width=400, height=240>
</p>

<p align="center">
  <a href="https://github.com/RektPunk/RektGBM/releases/latest">
    <img alt="Release" src="https://img.shields.io/github/v/release/RektPunk/rektgbm.svg">
  </a>
  <a href="https://github.com/RektPunk/RektGBM/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/RektPunk/RektGBM.svg">
  </a>
  <a href="https://pypi.org/project/rektgbm">
    <img alt="Pythonv" src="https://img.shields.io/pypi/pyversions/rektgbm.svg?logo=python&logoColor=white">
  </a>
  <a href="https://github.com/RektPunk/RektGBM/actions/workflows/lint.yaml">
    <img alt="Lint" src="https://github.com/RektPunk/RektGBM/actions/workflows/lint.yaml/badge.svg?branch=main">
  </a>
</p>

Are you sick and tired of the endless cycle of model comparisons and hyperparameter tuning? Meet **RektGBM**, your ultimate no-brainer solution to building good enough performance machine learning models. **RektGBM** seamlessly combines [LightGBM](https://github.com/microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost), two leading gradient boosting frameworks available today.

**RektGBM** simplifies the complexities of model building and optimization, delivering a robust, yet easy-to-use platform that empowers you to achieve superior results without the hassle. With the integration of [Optuna](https://github.com/optuna/optuna), a cutting-edge hyperparameter optimization framework, **RektGBM** fine-tunes your models to perfection, ensuring optimal performance with minimal effort.


# Installation
Getting started with RektGBM is a breeze. Install it using pip:
```bash
pip install rektgbm
```

# Key Features
- **RektDataset**: A wrapper for datasets used in `RektGBM` and `RektOptimizer`.
- **RektGBM**: A versatile gradient boosting model that leverages the power of both LightGBM and XGBoost.
- **RektOptimizer**: An Optuna-based optimizer that fine-tunes RektGBM hyperparameters for maximum efficiency.

# Usage
## Example workflow
Here’s a quick example to showcase how you can use RektGBM in your machine learning pipeline:

```python
from rektgbm import RektDataset, RektGBM, RektOptimizer
...
# Prepare your datasets
dtrain = RektDataset(data=X_train, label=y_train)
dtest = RektDataset(data=X_test, label=y_test)

# Initialize RektOptimizer to automatically detect task type, objective, and metric
rekt_optimizer = RektOptimizer()

# Optimize hyperparameters over 100 trials
rekt_optimizer.optimize_params(dataset=dtrain, n_trials=100)

# Display the best hyperparameters found
print(rekt_optimizer.best_params)

# Initialize RektGBM with the optimized hyperparameters
rekt_gbm = RektGBM(**rekt_optimizer.best_params)

# Train the model
rekt_gbm.fit(dataset=dtrain)

# Make predictions on the test set
preds = rekt_gbm.predict(dataset=dtest)

# DONE!
```

# More Examples
For more detailed examples, check out the [Examples](https://github.com/RektPunk/RektGBM/tree/main/examples) section.

# Contribution
Welcome all contributions! Please note that this project might have some bugs, so your contributions in identifying and fixing them are especially valuable! Whether you want to create an issue or submit a pull request, your input is appreciated. Feel free to get involved in any way that suits you.
