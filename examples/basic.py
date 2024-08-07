from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from rektgbm import RektDataset, RektGBM, RektOptimizer

X, y = make_regression(n_samples=10_000, n_features=10, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
dtrain = RektDataset(data=X_train, label=y_train)
dtest = RektDataset(data=X_test, label=y_test)

rekt_optimizer = RektOptimizer(method="both")
rekt_optimizer.optimize_params(
    dataset=dtrain,
    n_trials=10,
)
print(rekt_optimizer.best_params)

rekt_gbm = RektGBM(**rekt_optimizer.best_params)
rekt_gbm.fit(dataset=dtrain)
preds = rekt_gbm.predict(dataset=dtest)
