import streamlit as st
import warnings

from Pages.test_page import TestPage

warnings.filterwarnings("ignore")

from Pages.model_comparison_page import ModelComparisonPage
from Pages.datasets_page import DatasetsPage
from Pages.introduction_page import IntroductionPage
from Pages.association_rules import AssociationRulesPage
from Pages.feature_select_page import FeatureSelectPage
from Pages.association_rule_analysis_page import AssociationRuleAnalysisPage

if __name__ == '__main__':
    st.sidebar.title("Feature Selection with Association Rules")
    menu = st.sidebar.radio('Navigation', ('Introduction', "Datasets", "Association Rules",
                                           "Association Rule Analysis", "Feature Selection", "Model Comparison", "Test"))
    st.sidebar.title("Details")
    st.sidebar.info(
        "Author: Zvi Berger and Ofir Nassimi")
    st.sidebar.info(
        "This Project analyzing associations rules on different datasets for feature selection")
    st.sidebar.info(
        "[Project Proposal](https://docs.google.com/document/d/1ZIQc4LTywLEE4cW4iO5XRoU99oeujnbDJJHZpqAo7mU/edit)")

    st.sidebar.info("[Report](https://drive.google.com/file/d/1Vl_iJLNBdscOHjPpyP8pDdZ5yhiT9ESg/view?usp=sharing)")
    st.sidebar.info("[Github](https://github.com/BergerZvika/Feature-Selection-with-Association-Rules)")
    st.sidebar.info("[Google Colab Notebook](https://colab.research.google.com/drive/1fnJf6zoGL4Sovv8LYbm-2WWChMSkc4ed?usp=sharing)")

    st.sidebar.title("Related Work")
    st.sidebar.info(
        "[Feature Selection Algorithm Based on Association Rules Mining Method](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5222899)")

    st.sidebar.info(
        "[A new feature selection method based on association rules for diagnosis of erythemato-squamous diseases](https://reader.elsevier.com/reader/sd/pii/S0957417409003728?token=A6C5E697E425EBF0104E836671C78B3C3742AA3373B0962D3C03A6AE8FF3B6ABE7EE94522A9E64400349D75989E28DE1&originRegion=eu-west-1&originCreation=20220221130014)")

    st.sidebar.title("Kaggle Dataset")
    st.sidebar.info(
        "[House Price](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv)")
    st.sidebar.info(
        "[Diamond](https://www.kaggle.com/datasets/shivam2503/diamonds)")
    st.sidebar.info(
        "[Avocado](https://www.kaggle.com/datasets/timmate/avocado-prices-2020)")
    st.sidebar.info(
        "[NBA Rookie](https://data.world/exercises/logistic-regression-exercise-1)")


    introduction = IntroductionPage()
    datasets = DatasetsPage()
    feature_select = FeatureSelectPage()
    association_rules = AssociationRulesPage()
    association_rule_analysis = AssociationRuleAnalysisPage()
    model_comparison = ModelComparisonPage()
    test = TestPage()

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

    if menu == 'Test':
        test.show_page()
