import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from Pages.page import Page
from config import Config
import pandas as pd
import seaborn as sns



class AssociationRuleAnalysisPage(Page):
    def show_page(self):
        st.write("""# Association Rule Analysis""")
        st.markdown("""In this page we analysis  """)
        st.write("""## Associatiom Rule Table""")
        st.markdown("""      """)



        sort = st.selectbox("Sort by:",
                            ["Default", "Support", "Confidence", "Lift", "Conviction",
                             "Length Right Rule", "Length Left Rule"])
        table = ""
        if sort == "Default":
            table = Config.association_rule_table
        if sort == "Support":
            table = Config.association_rule_table.sort_values('support', ascending=False)
        if sort == "Confidence":
            table = Config.association_rule_table.sort_values('confidence', ascending=False)
        if sort == "Lift":
            table = Config.association_rule_table.sort_values('lift', ascending=False)
        if sort == "Conviction":
            table = Config.association_rule_table.sort_values('conviction', ascending=False)
        if sort == "Length Right Rule":
            table = Config.association_rule_table.sort_values('len_r')
        if sort == "Length Left Rule":
            table = Config.association_rule_table.sort_values('len_l')

        st.write(table)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        daownload_table = convert_df(table)
        st.download_button(label="download table", data=daownload_table, file_name="association_rules_analysis.csv",
                           mime='text/csv')

        st.write("""## Select Predict Feature""")
        feature = st.selectbox("Select Feature:",
                            list(Config.database.columns))

        feature_table = table.copy()
        index = []
        for i in feature_table.index:
            sp = i.split('->')
            if feature not in sp[1]:
                index.append(i)

        # index = [a for a in feature_table.index if feature not in a]
        feature_remove = feature_table.copy()
        feature_remove = feature_remove.drop(index)



        sort2 = st.selectbox("Sort Table by:",
                            ["Default", "Support", "Confidence", "Lift", "Conviction",
                             "Length Right Rule", "Length Left Rule"])
        table2 = ""
        if sort2 == "Default":
            table2 = feature_remove
        if sort2 == "Support":
            table2 = feature_remove.sort_values('support', ascending=False)
        if sort2 == "Confidence":
            table2 = feature_remove.sort_values('confidence', ascending=False)
        if sort2 == "Lift":
            table2 = feature_remove.sort_values('lift', ascending=False)
        if sort2 == "Conviction":
            table2 = feature_remove.sort_values('conviction', ascending=False)
        if sort2 == "Length Right Rule":
            table2 = feature_remove.sort_values('len_r')
        if sort2 == "Length Left Rule":
            table2 = feature_remove.sort_values('len_l')

        st.success(len(table2))
        st.write(table2)
        Config.association_rule_analysis = table2

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        daownload_table_2 = convert_df(table2)
        st.download_button(label="download table", data=daownload_table_2, file_name="association_rules_analysis2.csv",
                           mime='text/csv')







