# Import necessary libraries
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from rektgbm import RektDataset, RektGBM, RektOptimizer

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=10_000, n_features=10, n_informative=5)

# Ensure that target values are non-negative
y = np.maximum(y, 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Create RektDataset objects for training and testing data
dtrain = RektDataset(data=X_train, label=y_train)
dtest = RektDataset(data=X_test, label=y_test, reference=dtrain)

# Initialize RektOptimizer with specified settings for regression task
rekt_optimizer = RektOptimizer(
    method="both",  # Method: options are both, lightgbm, xgboost
    task_type="regression",  # Type of task: regression
    objective="gamma",  # Objective function
    metric="gamma",  # Metric: options are gamma, gamma_deviance
)

# Optimize hyperparameters using the training dataset over a specified number of trials
rekt_optimizer.optimize_params(dataset=dtrain, n_trials=10)

# Print the best hyperparameters found during optimization
print(rekt_optimizer.best_params)

# Initialize RektGBM model with the best hyperparameters
rekt_gbm = RektGBM(**rekt_optimizer.best_params)

# Train the model using the training dataset
rekt_gbm.fit(dataset=dtrain)

# Predict on the test dataset using the trained model
preds = rekt_gbm.predict(dataset=dtest)
