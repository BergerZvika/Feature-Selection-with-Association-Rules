# data processing
import math
import pandas as pd
import numpy as np
import streamlit as st

#Patterns Mining
from efficient_apriori import apriori

import warnings

from Pages.model_comparison_page import ModelComparisonPage

warnings.filterwarnings("ignore")
import sys

Alpha = 2


from Pages.datasets_page import DatasetsPage
from Pages.introduction_page import IntroductionPage
from Pages.association_rules import AssociationRulesPage
from Pages.feature_select_page import FeatureSelectPage

# to run:
from Pages.association_rule_analysis_page import AssociationRuleAnalysisPage

if __name__ == '__main__':
    st.sidebar.title("Feature Selection with Association Rules")
    menu = st.sidebar.radio('Navigation', ('Introduction', "Datasets", "Association Rules",
                                           "Association Rule Analysis", "Feature Selection", "Model Comparison"))
    st.sidebar.title("Details")
    st.sidebar.info(
        "Author: Zvi Berger and Ofir Nassimi")
    st.sidebar.info(
        "This Project analyzing associations rules on different datasets for feature selection")
    st.sidebar.info(
        "[Project Proposal](https://docs.google.com/document/d/1ZIQc4LTywLEE4cW4iO5XRoU99oeujnbDJJHZpqAo7mU/edit)")

    st.sidebar.info("[Report]")
    st.sidebar.info("[Github]")

    st.sidebar.title("Related Work")
    st.sidebar.info(
        "[Analyzing students’ answers using association rule mining based on feature selection](https://jsju.org/index.php/journal/article/view/236)")
    st.sidebar.info(
        "[Feature Selection by Mining Optimized Association Rules based on Apriori Algorithm](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.695.1446&rep=rep1&type=pdf)")

    st.sidebar.title("Kaggle Dataset")
    st.sidebar.info(
        "[Busiest Airports by Passenger Traffic](https://www.kaggle.com/jonahmary17/airports)")
    st.sidebar.info(
        "[Uber Fares Dataset](https://www.kaggle.com/yasserh/uber-fares-dataset)")
    st.sidebar.info(
        "[Top 250s in IMDB](https://www.kaggle.com/ramjasmaurya/top-250s-in-imdb)")
    st.sidebar.info(
        "[Miami Housing Dataset](https://www.kaggle.com/deepcontractor/miami-housing-dataset)")


    introduction = IntroductionPage()
    datasets = DatasetsPage()
    feature_select = FeatureSelectPage()
    association_rules = AssociationRulesPage()
    association_rule_analysis = AssociationRuleAnalysisPage()
    model_comparison = ModelComparisonPage()

    if menu == 'Introduction':
        introduction.show_page()

    if menu == 'Datasets':
        datasets.show_page()

    if menu == "Association Rules":
        association_rules.show_page()

    if menu == "Association Rule Analysis":
        association_rule_analysis.show_page()

    if menu == 'Feature Selection':
        feature_select.show_page()

    if menu == 'Model Comparison':
        model_comparison.show_page()