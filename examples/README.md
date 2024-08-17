# RektGBM Examples

This directory contains example scripts demonstrating the usage of RektGBM.

## Bash
You can run the script directly from a terminal
```bash
$ rektgbm --help
Usage: rektgbm [OPTIONS] DATA_PATH TEST_DATA_PATH TARGET [RESULT_PATH] [N_TRIALS]
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path           TEXT           Path to the training data file.  [required]
│ *    test_data_path      TEXT           Path to the test data file.  [required]
│ *    target              TEXT           Name of the target column.  [required]
│      result_path         [RESULT_PATH]  Path to the prediction results. [default: predict.csv]
│      n_trials            [N_TRIALS]     Number of optimization trials. [default: 100]
╰───────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.
╰───────────────────────────────────────────────────────────────────────────────────────────────────╯

$ rektgbm train.csv test.csv target predict.csv 100
```

## Python Scripts
For more advanced usage, check out the following example scripts.
```bash
examples
├── classification
│   ├── binary_classfication.py
│   └── multiclass_classification.py
├── rank
│   ├── basic_rank.py
├── regression
│   ├── basic_regression.py
│   ├── gamma_regression.py
│   ├── huber_regression.py
│   └── quantile_regression.py
└── README.md
```
