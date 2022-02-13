from datetime import time

import streamlit as st
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from Pages.page import Page
from config import Config

def predict(data, model):
    x=2
    # model.fit(Config.X_train, Config.y_train.ravel())
    # new_data=Config.scaler_x.transform([data])
    # res = model.predict(new_data)
    # return [res]

class ModelComparisonPage(Page):
    def show_page(self):
        st.write("""# Model Comparison""")
        # st.write("""### Introduction""")
        st.markdown("""This page compare """)


        machine = st.selectbox("Choose Model:",
                                   ["Linear Regression", "SVR", "Decision Tree", "Random Forest", "Neural Network",
                                    "SGD", "KNN", "Passive Aggressive"])

            if st.button("Predict"):
                model = LinearRegression()
                acc, result = 0, 0
                if machine == "Linear Regression":
                    model = LinearRegression()
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

                data = Config.database

                for col in Config.feature_selection:
                    data = data[col]
                result = predict(data, model)
                # my_bar = st.progress(0)
                #
                # for percent_complete in range(100):
                #     time.sleep(0.1)
                #     my_bar.progress(percent_complete + 1)

                st.write("""### Predicted University Rating on the current profile :""")
                result = result[0]
                result = int("%.f" % int(result))

                # print(result)

                if result >= 4 and result <= 5:
                        st.success(str(result))
                elif result == 3:
                        st.warning(str(result))
                else:
                        st.error(str(result))