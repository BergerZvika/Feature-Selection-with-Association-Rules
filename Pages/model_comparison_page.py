import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from Pages.page import Page
from config import Config


def train_split(data):
    x = data.drop(Config.predict_feature, axis=1)
    y = data[Config.predict_feature]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def evaluation(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    return [mse, r2, [y_test, y_predict]]


class ModelComparisonPage(Page):
    def show_page(self):
        st.write("""# Model Comparison""")
        st.markdown("""This page compare """)

        machine = st.selectbox("Choose Model:",
                               ["Linear Regression", "Decision Tree", "Random Forest", "AdaBoost"
                                   , "XGBoost", "SVR", "Neural Network", "SGD", "KNN", "Passive Aggressive"])
        if st.button("Predict"):
            model = LinearRegression()
            if machine == "Linear Regression":
                model = LinearRegression()
            if machine == "XGBoost":
                model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
            if machine == "SVR":
                model = svm.SVR()
            if machine == "Decision Tree":
                model = DecisionTreeRegressor()
            if machine == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, max_depth=10)
            if machine == "Neural Network":
                model = MLPRegressor(hidden_layer_sizes=(30, 50, 30), activation='relu', solver='adam',
                                     batch_size='auto',
                                     learning_rate='invscaling', learning_rate_init=0.001, shuffle=True)
            if machine == "SGD":
                model = SGDRegressor()
            if machine == "KNN":
                model = KNeighborsRegressor()
            if machine == "Passive Aggressive":
                model = PassiveAggressiveRegressor()
            if machine == "AdaBoost":
                model = AdaBoostRegressor()

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

            x_train = data_enconder_train.drop(Config.predict_feature, axis=1)
            x_test = data_enconder_test.drop(Config.predict_feature, axis=1)
            y_train = data_enconder_train[Config.predict_feature]
            y_test = data_enconder_test[Config.predict_feature]
            # x_train, x_test, y_train, y_test = train_split(data_enconder_test)

            # df = data_enconder
            x_train_support = x_train[Config.feature_selection['support']]
            x_test_support = x_test[Config.feature_selection['support']]
            x_train_confidence = x_train[Config.feature_selection['confidence']]
            x_test_confidence = x_test[Config.feature_selection['confidence']]
            x_train_lift = x_train[Config.feature_selection['lift']]
            x_test_lift = x_test[Config.feature_selection['lift']]
            x_train_lift_distance = x_train[Config.feature_selection['lift_distance_to_1']]
            x_test_lift_distance = x_test[Config.feature_selection['lift_distance_to_1']]

            mse_data, r2_data, eva_data = evaluation(model, x_train, x_test, y_train, y_test)
            mse_support, r2_support, eva_support = evaluation(model, x_train_support, x_test_support, y_train, y_test)
            mse_confidence, r2_confidence, eva_confidence = evaluation(model, x_train_confidence, x_test_confidence, y_train, y_test)
            mse_lift, r2_lift, eva_lift = evaluation(model, x_train_lift, x_test_lift, y_train, y_test)
            mse_lift_distance, r2_lift_distance, eva_lift_distance = evaluation(model, x_train_lift_distance, x_test_lift_distance, y_train, y_test)

            st.write(f"### Comparison Table by {machine}""")

            r2 = [r2_data, r2_support, r2_confidence, r2_lift, r2_lift_distance]
            mse = [mse_data, mse_support, mse_confidence, mse_lift, mse_lift_distance]
            data_table = {'Features': ['All Features', 'Select feature by support', 'Select feature by confidence',
                                       'Select feature by lift', 'Select feature by distance lift to 1'],
                          'R-Square Score': r2,
                          'Mean Sqaure Error': mse,
                          }
            table = pd.DataFrame(data_table, columns=['Features', 'R-Square Score', 'Mean Sqaure Error'])

            st.table(table)

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            download_table = convert_df(table)
            st.download_button(label="download table", data=download_table, file_name="comparasion_model.csv",
                               mime='text/csv')

            # graph
            st.write(f'### {machine} Graphs')

            fig, ax = plt.subplots()
            plt.plot(eva_data[0], eva_data[1], "^", color='#EDB120')
            plt.ylabel("Actual")
            plt.xlabel("Predict")
            plt.title("Without Feature Selection")
            st.pyplot(fig)
            st.markdown("""""")

            fig, ax = plt.subplots()
            plt.plot(eva_support[0], eva_support[1], "^", color='#EDB120')
            plt.ylabel("Actual")
            plt.xlabel("Predict")
            plt.title("Feature Selection by support")
            st.pyplot(fig)
            st.markdown("""""")

            fig, ax = plt.subplots()
            plt.plot(eva_confidence[0], eva_confidence[1], "^", color='#EDB120')
            plt.ylabel("Actual")
            plt.xlabel("Predict")
            plt.title("Feature Selection by confidence")
            st.pyplot(fig)
            st.markdown("""""")

            fig, ax = plt.subplots()
            plt.plot(eva_lift[0], eva_lift[1], "^", color='#EDB120')
            plt.ylabel("Actual")
            plt.xlabel("Predict")
            plt.title("Feature Selection by lift")
            st.pyplot(fig)
            st.markdown("""""")

            fig, ax = plt.subplots()
            plt.plot(eva_lift_distance[0], eva_lift_distance[1], "^", color='#EDB120')
            plt.ylabel("Actual")
            plt.xlabel("Predict")
            plt.title("Feature Selection by lift distance to 1")
            st.pyplot(fig)
            st.markdown("""""")

