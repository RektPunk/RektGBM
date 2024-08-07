# Import necessary libraries from scikit-learn and rektgbm packages
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from rektgbm import RektDataset, RektGBM, RektOptimizer

# Generate a synthetic binary classification dataset
X, y = make_classification(
    n_samples=10_000, n_features=10, n_informative=5, n_classes=2
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Create RektDataset objects for training and testing data
dtrain = RektDataset(data=X_train, label=y_train)
dtest = RektDataset(data=X_test, label=y_test)

# Initialize RektOptimizer for automatic task type, objective, and metric detection
rekt_optimizer = RektOptimizer()

# Alternatively, manually select optimizer settings (commented out)
# rekt_optimizer = RektOptimizer(
#     method="lightgbm",  # Optimization method: options are both, lightgbm, xgboost
#     task_type="binary", # Type of task: binary
#     objective="binary", # Objective function
#     metric = "auc",     # Metric: options are logloss and auc
# )

# Optimize hyperparameters using the training dataset over a specified number of trials
rekt_optimizer.optimize_params(dataset=dtrain, n_trials=10)

# Print the best hyperparameters found during optimization
print(rekt_optimizer.best_params)

# Initialize RektGBM model with the best hyperparameters
rekt_gbm = RektGBM(**rekt_optimizer.best_params)

# Train the model using the training dataset
rekt_gbm.fit(dataset=dtrain)

# Predict on the test dataset using the trained model
preds = rekt_gbm.predict(RektDataset(X_test, y_train))
