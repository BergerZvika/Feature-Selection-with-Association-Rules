import pandas as pd
import streamlit as st
from efficient_apriori import apriori

from Pages.page import Page
from config import Config


class AssociationRulesPage(Page):
    def show_page(self):
        st.write("""# Association Rules """)
        st.markdown("""This page find association rules on dataset with apriori algorithm""")
        st.markdown("""  """)
        st.markdown("""## Find Association Rules on your Data""")
        st.write("""#### Dataset""")
        data = st.selectbox("Choose Dataset:",
                                   ["House Sale Price", "Imdb Movies", "Uber", "Busiest Airports"])

        dataset = ""
        if data == "House Sale Price":
                dataset = Config.house_data
        if data == "Imdb Movies":
                dataset = Config.imdb_data
        if data == "Uber":
                dataset = Config.uber_data
        if data == "Busiest Airports":
                dataset = Config.airports_data

        st.write("""#### Association Rules""")
        support = st.number_input("Choose support threshold (0 - 1):", value=0.1)

        if support > 1:
                st.error("Your support bigger than 1")
        if support < 0:
                st.error("Your support less than 0")

        confidence = st.number_input("Choose confidence threshold (0 - 1):", value=0.6)

        if confidence > 1:
                st.error("Your confidence bigger than 1")
        if confidence < 0:
                st.error("Your confidence less than 0")

        if st.button("Find Association Rules"):
                dataset_rules = dataset.copy()
                # Defining numeric and categorical columns
                numeric_columns = dataset_rules.dtypes[(dataset_rules.dtypes == "float64") | (dataset_rules.dtypes == "int64")].index.tolist()
                very_numerical = [nc for nc in numeric_columns if dataset_rules[nc].nunique() > 20]

                # Binning all numeric columns in the same manner:
                for c in very_numerical:
                        try:
                                dataset_rules[c] = pd.qcut(dataset_rules[c], 5, labels=["very low", "low", "medium", "high", "very high"])
                        except:
                                # sometimes for highly skewed data, we cannot perform qcut as most quantiles are equal
                                dataset_rules[c] = pd.cut(dataset_rules[c], 5, labels=["very low", "low", "medium", "high", "very high"])

                if data == "House Sale Price":
                    good_columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'BldgType',
                                    'LotArea', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'LotFrontage', 'TotalBsmtSF',
                                    'SalePrice']
                    dataset_rules = dataset_rules[good_columns]

                records = dataset_rules.to_dict(orient='records')
                transactions = []
                for r in records:
                        transactions.append(list(r.items()))

                # Rules mining process:
                itemsets, rules = apriori(transactions, min_support=support, min_confidence=confidence,
                                          output_transaction_ids=False)

                if len(rules) == 0:
                    st.error("number of rules: " + str(len(rules)))
                else:
                    st.success("number of rules: " + str(len(rules)))

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

                    Config.association_rule_table = rules_df
                    Config.database = dataset

                    st.write(rules_df)

                    @st.cache
                    def convert_df(df):
                        # IMPORTANT: Cache the conversion to prevent computation on every rerun
                        return df.to_csv().encode('utf-8')

                    daownload_table = convert_df(rules_df)
                    st.download_button(label="download table", data=daownload_table, file_name="association_rules.csv", mime='text/csv')
