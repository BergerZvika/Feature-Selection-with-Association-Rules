import pandas as pd
import streamlit as st


from Pages.page import Page
from config import Config


class FeatureSelectPage(Page):
    def show_page(self):
            st.write("""# Feature Selection""")
            st.markdown("""In this page you can choose k to find the top k features by support, confidence and lift.""")

            st.write("""#### Find Top k features""")
            st.write(Config.association_rule_analysis)

            k = st.number_input("Insert k: ", 1, 20, 5, 1)
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
                            i += 2
                    return list(feature)

                support_feature = feature_selection(Config.association_rule_analysis.sort_values('support', ascending=False))
                confidence_feature = feature_selection(Config.association_rule_analysis.sort_values('confidence', ascending=False))
                lift_feature = feature_selection(Config.association_rule_analysis.sort_values('lift', ascending=False))

                index = 0
                for i in Config.association_rule_analysis.index:
                    Config.association_rule_analysis.at[i, 'lift'] = abs(Config.association_rule_analysis.iloc[index]['lift'] - 1)
                    index += 1

                lift_distance_feature = feature_selection(Config.association_rule_analysis.sort_values('lift', ascending=False))

                st.write("""### Top k features""")
                feature_table = {'support' : support_feature,
                      'confidence' : confidence_feature,
                      'lift': lift_feature,
                        'lift_distance_to_1': lift_distance_feature}

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
