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
    house_data = pd.read_csv('data/train.csv', index_col='Id')
    imdb_data = pd.read_csv('data/CarPrice_Assignment.csv')
    uber_data = pd.read_csv('data/uber.csv')
    airports_data = pd.read_csv('data/busiestAirports.csv')

    house_data = fillna(house_data)
    imdb_data = fillna(imdb_data)
    uber_data = fillna(uber_data)
    airports_data = fillna(airports_data)

    database = house_data
    association_rule_table = "Not Found Association Rule Table!!!"
    association_rule_analysis = "Not Found Association Rule Analysis Table!!!"
    feature_selection = "Not Found Feature Selection"
    predict_feature = "Not Found Predict Feature"
    test = "Not Found Test"

    file = open("test.txt", 'w')
    file.close()
    os.remove("test.txt")

