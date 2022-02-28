import os

import pandas as pd


def fillna(dataset):
    # Defining numeric and categorical columns
    numeric_columns = dataset.dtypes[(dataset.dtypes == "float64") | (dataset.dtypes == "int64")].index.tolist()
    very_numerical = [nc for nc in numeric_columns if dataset[nc].nunique() > 20]
    categorical_columns = [c for c in dataset.columns if c not in numeric_columns]
    ordinals = list(set(numeric_columns) - set(very_numerical))

    # Filling Null Values with the column's mean
    na_columns = dataset[very_numerical].isna().sum()
    na_columns = na_columns[na_columns > 0]
    for nc in na_columns.index:
        dataset[nc].fillna(dataset[nc].mean(), inplace=True)

    # Dropping and filling NA values for categorical columns:
    nul_cols = dataset[categorical_columns].isna().sum() / len(dataset)
    drop_us = nul_cols[nul_cols > 0.7]
    dataset = dataset.drop(drop_us.index, axis=1)
    categorical_columns = list(set(categorical_columns) - set(drop_us.index))
    dataset[categorical_columns] = dataset[categorical_columns].fillna('na')
    return dataset


class Config:
    # read the database
    DATASET_1 = "House Sale Price"
    DATASET_2 = "NBA Rookie"
    DATASET_3 = "Diamods Price"
    DATASET_4 = "Trains in France"
    datasets_names = [DATASET_1, DATASET_2, DATASET_3, DATASET_4]

    dataset_1 = pd.read_csv('data/train.csv', index_col=[0])
    dataset_2 = pd.read_csv('data/nba_rookie.csv', index_col=[0])
    dataset_3 = pd.read_csv('data/diamonds.csv', index_col=[0])
    dataset_4 = pd.read_csv('data/transport.csv', index_col=[0])

    dataset_1 = fillna(dataset_1)
    dataset_2 = fillna(dataset_2)
    dataset_3 = fillna(dataset_3)
    dataset_4 = fillna(dataset_4)
    datasets = {
        DATASET_1: dataset_1,
        DATASET_2: dataset_2,
        DATASET_3: dataset_3,
        DATASET_4: dataset_4
    }

    database = dataset_1
    association_rule_table = "Not Found Association Rule Table!!!"
    association_rule_analysis = "Not Found Association Rule Analysis Table!!!"
    feature_selection = "Not Found Feature Selection"
    predict_feature = "Not Found Predict Feature"
    test = "Not Found Test"

    file = open("test.txt", 'w')
    file.close()
    os.remove("test.txt")

