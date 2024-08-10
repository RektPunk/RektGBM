# Import necessary libraries
import numpy as np
import pandas as pd

from rektgbm import RektDataset, RektGBM, RektOptimizer

# Generate a synthetic dataset
# 'query_id' simulates groups of queries, and 'relevance' indicates the relevance of the item to the query.
df = pd.DataFrame(
    {
        "query_id": [
            i for i in range(1_000) for j in range(10)
        ],  # 1000 unique queries, each with 10 items
        "var1": np.random.random(size=(10_000,)),  # Random feature 1
        "var2": np.random.random(size=(10_000,)),  # Random feature 2
        "var3": np.random.random(size=(10_000,)),  # Random feature 3
        "relevance": list(np.random.permutation([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]))
        * 1_000,  # Random relevance scores
    }
)

# Generate a test dataset for later evaluation
X_test = pd.DataFrame(
    {
        "var1": np.random.random(size=(1_000,)),  # Random feature 1
        "var2": np.random.random(size=(1_000,)),  # Random feature 2
        "var3": np.random.random(size=(1_000,)),  # Random feature 3
    }
)

# Split the dataset into training (80%) and validation (20%) sets
train_df = df[:8000]  # First 80% of the data
validation_df = df[8000:]  # Remaining 20% of the data

# Grouping for the ranking task (required for rank objective)
query_ids_train = train_df.groupby("query_id")["query_id"].count().to_numpy()
X_train = train_df.drop(["query_id", "relevance"], axis=1)  # Training features
y_train = train_df["relevance"]  # Training labels (relevance scores)

query_ids_validation = validation_df.groupby("query_id")["query_id"].count().to_numpy()
X_validation = validation_df.drop(
    ["query_id", "relevance"], axis=1
)  # Validation features
y_validation = validation_df["relevance"]  # Validation labels (relevance scores)

# Create RektDataset objects for training and validation
dtrain = RektDataset(data=X_train, label=y_train, group=query_ids_train)
dvalid = RektDataset(data=X_validation, label=y_validation, group=query_ids_validation)
dtest = RektDataset(data=X_test)  # Test dataset does not require group information


# Initialize RektOptimizer for automatic task type, objective, and metric detection
rekt_optimizer = RektOptimizer()

# Alternatively, manually select optimizer settings (commented out)
# rekt_optimizer = RektOptimizer(
#     method="both",      # Method: options are both (default), lightgbm, xgboost
#     task_type="rank",   # Type of task: rank
#     objective="ndcg",   # Objective function: options are lambdarank, ndcg
#     metric="map",       # Metric: options are ndcg, map
#     additional_params={
#         "eval_at": 3    # Evaluate model performance at the top 3 ranks, default 5
#     }
# )

# Optimize model hyperparameters using the training and validation datasets
rekt_optimizer.optimize_params(
    dataset=dtrain,
    valid_set=dvalid,  # Validation set is necessary for ranking tasks
    n_trials=10,  # Number of optimization trials (for demonstration; usually, more trials are preferred)
)

# Print the best hyperparameters found during optimization
print(rekt_optimizer.best_params)

# Initialize RektGBM model with the best hyperparameters
rekt_gbm = RektGBM(**rekt_optimizer.best_params)

# Train the model on the training dataset and validate using the validation set
rekt_gbm.fit(dataset=dtrain, valid_set=dvalid)

# Predict on the test dataset using the trained model
preds = rekt_gbm.predict(dataset=dtest)
