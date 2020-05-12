# Create K-folds in the training set
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def create_K_folds(train_csv_file, n_splits):
    df = pd.read_csv(Path / train_csv_file)

    # Create a column in df names "kfold"
    df.loc[:, "kfold"] = -1
    # Change the X and y according to logic
    X, y = None, None
    skf = StratifiedKFold(n_splits = n_splits)

    for fold, (train_idx, validation_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[validation_idx], y[validation_idx]

        df.loc[validation_idx, "kfold"] = fold
    
    # So basically I can either save the folds
    # Save the final changed df
    df.to_csv(Path / "train_folds.csv") 


if __name__ == "__main__":
    filename = ""
    Path = ""
    create_K_folds(filename, 10)