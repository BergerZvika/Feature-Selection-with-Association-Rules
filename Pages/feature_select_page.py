import time
import pandas as pd
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
    model.fit(Config.X_train, Config.y_train.ravel())
    new_data=Config.scaler_x.transform([data])
    res = model.predict(new_data)
    return [res]

class FeatureSelectPage(Page):
    def show_page(self):
            st.write("""# Feature Selectection""")
            st.markdown("""In this page  """)

            st.write("""#### Find Top k features""")
            st.write(Config.association_rule_analysis)

            k = st.number_input("Insert k: ", 1, 10, 5, 1)
            if st.button("Find top k Feature"):
                def feature_selection(data):
                    feature = set()
                    for i in data.index:
                        sp = i.split('->')
                        left = sp[0]
                        left = left.replace('{', '').replace('}', '').replace(' ', '').replace('(', '').replace('\'', '')
                        left = left.split(',')
                        i = 0
                        while i < len(left) and len(feature) < k:
                            feature.add(left[i])
                            i+= 2

                    # st.write(feature)
                    # if len(feature) < k:
                    #     st.error("k bigger than length of feature")

                    return list(feature)

                support_feature = feature_selection(Config.association_rule_analysis.sort_values('support', ascending=False))
                confidence_feature = feature_selection(Config.association_rule_analysis.sort_values('confidence', ascending=False))
                lift_feature = feature_selection(Config.association_rule_analysis.sort_values('lift', ascending=False))

                st.markdown("""Top k feature """)
                feature_table = {'support' : support_feature,
                      'confidence' : confidence_feature,
                      'lift': lift_feature}
                df = pd.DataFrame(data=feature_table)
                st.write(df)

                Config.feature_selection = feature_table

                @st.cache
                def convert_df(data):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return data.to_csv().encode('utf-8')

                daownload_table = convert_df(df)
                st.download_button(label="download feature table", data=daownload_table, file_name="feature_table.csv",
                                   mime='text/csv')
