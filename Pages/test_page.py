import os
from efficient_apriori import apriori
import pandas as pd
from pathlib import Path
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


from Pages.page import Page
from config import Config



def save_table(df, directory_path, file_name):
    filepath = Path(f'{directory_path}/{file_name}')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

def find_rules(support, confidence, transactions):
    itemsets, rules = apriori(transactions, min_support=support, min_confidence=confidence,
                              output_transaction_ids=False)

    # Analyzing the rules:
    if len(rules) > 0:
        attrs = [a for a in dir(rules[0]) if not a.startswith("_")]
        rules_rec = []
        for r in rules:
            rdict = {}
            for a in attrs:
                rdict[a] = getattr(r, a)
                rdict["rule"] = str(r).split("} (")[0] + "}"
                rdict["len_l"] = len(r.lhs)
                rdict["len_r"] = len(r.rhs)
            rules_rec.append(rdict)

        rules_df = pd.DataFrame(rules_rec)
        rules_df.set_index('rule', inplace=True)
        rules_df = rules_df[
            ['len_l', 'len_r', 'count_lhs', 'count_rhs', 'support', 'confidence', 'lift', 'rpf',
             'conviction']]
        st.write(f"Finished to find associations rules with support={support}.")
        Config.file.write(f'{len(rules)} rules with support={support} and confidence={confidence} were found.\n')
        return rules_df

def anylisis_rules(df, predict_feature, support):
    feature_table = df.copy()
    index = []
    for i in feature_table.index:
        sp = i.split('->')
        if predict_feature not in sp[1]:
            index.append(i)

    feature_remove = feature_table.copy()
    feature_remove = feature_remove.drop(index)
    st.write(f'Association rules analysis process for support={support} is done.')
    Config.file.write(f'{len(feature_remove)} rules with correlation to {predict_feature} were found.\n')
    return feature_remove

def select_feature(df, k, support):
    def feature_selection(data):
        feature = set()
        for i in data.index:
            sp = i.split('->')
            left = sp[0]
            left = left.replace('{', '').replace('}', '').replace('(', '').replace('\'', '')
            left = left.split(',')
            i = 0
            while i < len(left) and len(feature) < k:
                f = left[i].strip()
                feature.add(f)
                feature.add(left[i])
                i += 2
        return list(feature)

    support_feature = feature_selection(df.sort_values('support', ascending=False))
    confidence_feature = feature_selection(df.sort_values('confidence', ascending=False))
    lift_feature = feature_selection(df.sort_values('lift', ascending=False))

    index = 0
    for i in df.index:
        df.at[i, 'lift'] = abs(df.iloc[index]['lift'] - 1)
        index += 1

    lift_distance_feature = feature_selection(df.sort_values('lift', ascending=False))

    feature_table = {'support': support_feature,
                     'confidence': confidence_feature,
                     'lift': lift_feature,
                     'lift_distance_to_1': lift_distance_feature}

    data = pd.DataFrame(data=feature_table)
    st.write(f'Feature selection for k={k} and support={support} is done.')
    return data

def evaluation(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    return [mse, r2, [y_test, y_predict]]

def models_comparison(directory, features, predict_feature, support, k):
    numeric_columns = Config.database.dtypes[
        (Config.database.dtypes == "float64") | (Config.database.dtypes == "int64")].index.tolist()
    categorical_columns = [c for c in Config.database.columns if c not in numeric_columns]

    data_enconder_train = Config.database.copy()
    labelencoder = LabelEncoder()
    for c in categorical_columns:
        data_enconder_train[c] = labelencoder.fit_transform(data_enconder_train[c])

    data_enconder_test = Config.test.copy()
    labelencoder = LabelEncoder()
    for c in categorical_columns:
        data_enconder_test[c] = labelencoder.fit_transform(data_enconder_test[c])

    x_train = data_enconder_train.drop(predict_feature, axis=1)
    x_test = data_enconder_test.drop(predict_feature, axis=1)
    y_train = data_enconder_train[predict_feature]
    y_test = data_enconder_test[predict_feature]

    x_train_support = x_train[features['support']]
    x_test_support = x_test[features['support']]
    x_train_confidence = x_train[features['confidence']]
    x_test_confidence = x_test[features['confidence']]
    x_train_lift = x_train[features['lift']]
    x_test_lift = x_test[features['lift']]
    x_train_lift_distance = x_train[features['lift_distance_to_1']]
    x_test_lift_distance = x_test[features['lift_distance_to_1']]

    models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(n_estimators=100, max_depth=10),
              AdaBoostRegressor(), xgb.XGBRegressor(objective="reg:linear", random_state=42)]
    i = 0
    machine = ""
    for model in models:
        if i == 0:
            machine = "Linear Regression"
        if i == 1:
            machine = "Decision Tree"
        if i == 2:
            machine = "Random Forest"
        if i == 3:
            machine = "Ada Boost"
        if i == 4:
            machine = "XGBoost"

        try:
            path = Path(f'{directory}/{machine}')
            os.mkdir(path, 0o666)
        except:
            path = Path(f'{directory}/{machine}')

        mse_data, r2_data, eva_data = evaluation(model, x_train, x_test, y_train, y_test)
        mse_support, r2_support, eva_support = evaluation(model, x_train_support, x_test_support, y_train, y_test)
        mse_confidence, r2_confidence, eva_confidence = evaluation(model, x_train_confidence, x_test_confidence,
                                                                   y_train, y_test)
        mse_lift, r2_lift, eva_lift = evaluation(model, x_train_lift, x_test_lift, y_train, y_test)
        mse_lift_distance, r2_lift_distance, eva_lift_distance = evaluation(model, x_train_lift_distance,
                                                                            x_test_lift_distance, y_train, y_test)


        r2 = [r2_data, r2_support, r2_confidence, r2_lift, r2_lift_distance]
        mse = [mse_data, mse_support, mse_confidence, mse_lift, mse_lift_distance]
        data_table = {'Features': ['All Features', 'Select feature by support', 'Select feature by confidence',
                                   'Select feature by lift', 'Select feature by distance lift to 1'],
                      'R-Square Score': r2,
                      'Mean Sqaure Error': mse,
                      }
        table = pd.DataFrame(data_table, columns=['Features', 'R-Square Score', 'Mean Sqaure Error'])
        save_table(table, path, "models_comparison.csv")

        plt.plot(eva_data[0], eva_data[1], "^", color='#EDB120')
        plt.ylabel("Actual")
        plt.xlabel("Predict")
        plt.title("Without Feature Selection")
        image = f"{path}/Without_Feature_Selection.png"
        plt.savefig(image)

        plt.plot(eva_support[0], eva_support[1], "^", color='#EDB120')
        plt.ylabel("Actual")
        plt.xlabel("Predict")
        plt.title("Feature Selection by support")
        image = f"{path}/support.png"
        plt.savefig(image)

        plt.plot(eva_confidence[0], eva_confidence[1], "^", color='#EDB120')
        plt.ylabel("Actual")
        plt.xlabel("Predict")
        plt.title("Feature Selection by confidence")
        image = f"{path}/confidence.png"
        plt.savefig(image)

        plt.plot(eva_lift[0], eva_lift[1], "^", color='#EDB120')
        plt.ylabel("Actual")
        plt.xlabel("Predict")
        plt.title("Feature Selection by lift")
        image = f"{path}/lift.png"
        plt.savefig(image)

        plt.plot(eva_lift_distance[0], eva_lift_distance[1], "^", color='#EDB120')
        plt.ylabel("Actual")
        plt.xlabel("Predict")
        plt.title("Feature Selection by lift distance to 1")
        image = f"{path}/lift_distance_to_1.png"
        plt.savefig(image)

        i += 1
    st.write(f'Models comparison for k={k} and support={support} is done.')

class TestPage(Page):
    def show_page(self):
        try:
            directory = Path(f'./tests')
            os.mkdir(directory, 0o666)
        except:
            directory = Path(f'./tests')
        st.write("""# Test Page """)
        st.markdown("""In this test page, you can make tests on the datasets and find association rules, analyze them, 
        do feature selection by different top k and compare models. First you need to choose a dataset and the number 
        of tests you want. Each test finds association rules with 3 constant support thresholds - 0.1, 0.05, 0.01 and 
        analyzes them. In the next step, The program do the feature selection process for 4 constant k - 3, 5, 7, 10. 
        In the last step, the models are compared.""")
        st.markdown("""Every test is saved under a folder named by the dataset name, which is located inside "tests" 
        folder""")
        st.write("""### Test results:""")
        st.markdown("""
        After all the tests are done, every test folder contains the following:
        * Statistics text file - shows these statistics:
            * Number of rules found for each support
            * Number of rules correlated to the selected feature
        * Train csv file - all the data that used t train the model
        * Test csv file - all the data that used to test the model
        * 3 supports folders - one for each support, contains 4 folders for each k and these folders contain folders 
            for each model showed in class. Every model folder contains:
            * Model without feature selection graph
            * Model with feature selection by support graph
            * Model with feature selection by confidence graph
            * Model with feature selection by lift graph
            * Model with feature selection by lift distance to 1 graph
            * Models comparison csv file of R-square error and mean square error
        """)

        data = st.selectbox("Choose Dataset:",
                            Config.datasets_names)

        df = Config.datasets[data]
        length = len(df)
        dataset = Config.datasets[data].sample(frac=1).reset_index(drop=True)
        Config.test = dataset[int(length / 5 * 4):]
        dataset = dataset[:int(length / 5 * 4)]

        # df = {}
        # dataset = {}
        # if data == Config.DATASET_1:
        #         df = Config.dataset_1
        #         dataset = Config.dataset_1.sample(frac=1).reset_index(drop=True)
        #         Config.test = dataset[int((len(Config.dataset_1) / 5) * 4):]
        #         dataset = dataset[:int((len(Config.dataset_1) / 5) * 4)]
        # if data == Config.DATASET_2:
        #         df = Config.dataset_2
        #         dataset = Config.dataset_2.sample(frac=1).reset_index(drop=True)
        #         Config.test = dataset[int((len(Config.dataset_2) / 5) * 4):]
        #         dataset = dataset[:int((len(Config.dataset_2) / 5) * 4)]
        # if data == Config.DATASET_3:
        #         df = Config.dataset_3
        #         dataset = Config.dataset_3.sample(frac=1).reset_index(drop=True)
        #         Config.test = dataset[int((len(Config.dataset_3) / 5) * 4):]
        #         dataset = dataset[:int((len(Config.dataset_3) / 5) * 4)]
        # if data == Config.DATASET_4:
        #         df = Config.dataset_4
        #         dataset = Config.dataset_4.sample(frac=1).reset_index(drop=True)
        #         Config.test = dataset[int((len(Config.dataset_4) / 5) * 4):]
        #         dataset = dataset[:int((len(Config.dataset_4) / 5) * 4)]

        iterations = st.number_input("Choose number of tests (1-20):", value=3)
        if iterations > 20:
            st.error("Too many tests")
        if iterations < 0:
            st.error("Number of tests is less than 1")

        if st.button("Test"):
            st.write("Start testing. This may take a few minutes..")
            try:
                directory_path = Path(f'./tests/{data}')
                os.mkdir(directory_path, 0o666)
            except:
                directory_path = Path(f'./tests/{data}')
            save_table(df, directory_path, f'{data}.csv')

            for i in range(iterations):
                st.write(f'Start test number {i + 1}')
                try:
                    directory_path = Path(f'./tests/{data}/test{i+1}')
                    os.mkdir(directory_path, 0o666)
                except:
                    directory_path = Path(f'./tests/{data}/test{i+1}')

                Config.file = open(f'{directory_path}/statistics.txt', 'w')
                Config.file.write(f'{data} Test {i+1}\n\n')

                save_table(dataset, directory_path, 'train.csv')
                save_table(Config.test, directory_path, 'test.csv')

                #### associations rules fund
                dataset_rules = dataset.copy()
                # Defining numeric and categorical columns
                numeric_columns = dataset_rules.dtypes[
                    (dataset_rules.dtypes == "float64") | (dataset_rules.dtypes == "int64")].index.tolist()
                very_numerical = [nc for nc in numeric_columns if dataset_rules[nc].nunique() > 20]

                # Binning all numeric columns in the same manner:
                for c in very_numerical:
                    try:
                        dataset_rules[c] = pd.qcut(dataset_rules[c], 5,
                                                   labels=["very low", "low", "medium", "high", "very high"])
                    except:
                        # sometimes for highly skewed data, we cannot perform qcut as most quantiles are equal
                        dataset_rules[c] = pd.cut(dataset_rules[c], 5,
                                                  labels=["very low", "low", "medium", "high", "very high"])

                if data == "House Sale Price":
                    good_columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond',
                                    'BldgType',
                                    'LotArea', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'LotFrontage', 'TotalBsmtSF',
                                    'SalePrice']
                    dataset_rules = dataset_rules[good_columns]

                records = dataset_rules.to_dict(orient='records')
                transactions = []
                for r in records:
                    transactions.append(list(r.items()))

                try:
                    directory_support_1 = Path(f'./tests/{data}/test{i+1}/support 0.1')
                    os.mkdir(directory_support_1, 0o666)
                except:
                    directory_support_1 = Path(f'./tests/{data}/test{i+1}/support 0.1')
                rules_1 = find_rules(support=0.1, confidence=0.6, transactions=transactions)
                save_table(rules_1, directory_support_1, 'associations_rules.csv')

                try:
                    directory_support_05 = Path(f'./tests/{data}/test{i+1}/support 0.05')
                    os.mkdir(directory_support_05, 0o666)
                except:
                    directory_support_05 = Path(f'./tests/{data}/test{i+1}/support 0.05')
                rules_2 = find_rules(support=0.05, confidence=0.6, transactions=transactions)
                save_table(rules_2, directory_support_05, 'associations_rules.csv')

                try:
                    directory_support_01 = Path(f'./tests/{data}/test{i+1}/support 0.01')
                    os.mkdir(directory_support_01, 0o666)
                except:
                    directory_support_01 = Path(f'./tests/{data}/test{i+1}/support 0.01')
                rules_3 = find_rules(support=0.01, confidence=0.6, transactions=transactions)
                save_table(rules_3, directory_support_01, 'associations_rules.csv')
                Config.file.write('\n')

                #### associations rules analysis
                l = list(Config.database.columns)
                l.reverse()
                Config.predict_feature = l[0]

                anylisis_table_1 = anylisis_rules(rules_1, Config.predict_feature, 0.1)
                save_table(anylisis_table_1, directory_support_1, 'associations_rules_analysis.csv')
                anylisis_table_2 = anylisis_rules(rules_2, Config.predict_feature, 0.05)
                save_table(anylisis_table_2, directory_support_05, 'associations_rules_analysis.csv')
                anylisis_table_3 = anylisis_rules(rules_3, Config.predict_feature, 0.01)
                save_table(anylisis_table_3, directory_support_01, 'associations_rules_analysis.csv')
                Config.file.write('\n')

                ### feature select
                K = [3, 5, 7, 10]
                for k in K:
                    try:
                        path1 = Path(f'{directory_support_1}/k={k}')
                        os.mkdir(path1, 0o666)
                    except:
                        path1 = Path(f'{directory_support_1}/k={k}')
                    feature_table_1_k = select_feature(anylisis_table_1, k, 0.1)
                    save_table(feature_table_1_k, path1, f'select_feature_table_k={k}.csv')
                    try:
                        path2 = Path(f'{directory_support_05}/k={k}')
                        os.mkdir(path2, 0o666)
                    except:
                        path2 = Path(f'{directory_support_05}/k={k}')
                    feature_table_05_k = select_feature(anylisis_table_2, k, 0.05)
                    save_table(feature_table_05_k, path2, f'select_feature_table_k={k}.csv')
                    try:
                        path3 = Path(f'{directory_support_01}/k={k}')
                        os.mkdir(path3, 0o666)
                    except:
                        path3 = Path(f'{directory_support_01}/k={k}')
                    feature_table_01_k = select_feature(anylisis_table_3, k, 0.01)
                    save_table(feature_table_01_k, path3, f'select_feature_table_k={k}.csv')

                    ### models comparison
                    models_comparison(path1, feature_table_1_k, Config.predict_feature, 0.1, k)
                    models_comparison(path2, feature_table_05_k, Config.predict_feature, 0.05, k)
                    models_comparison(path3, feature_table_01_k, Config.predict_feature, 0.01, k)


            st.write("Done")
            Config.file.close()



