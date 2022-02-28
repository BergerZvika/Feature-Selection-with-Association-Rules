import streamlit as st

from Pages.page import Page


class IntroductionPage(Page):
    def show_page(self):
        st.write("""# Tabular Data Science Final Project""")
        st.write("""## Feature Selection with Association Rules""")
        st.write("""### Introduction""")
        st.markdown("""Data science usually contains a big 
        amount of data - values and features. In order to make 
        it more efficient, we can do feature selection. One way 
        to do it is by using association rules. In my project, I’d 
        like to determine if a single feature is important by using 
        the association rules we’ve seen in class and to predict the 
        most important ones.""")

        st.write("""### Our Solution""")
        st.markdown("""My idea to try to solve this problem is by using
         the concepts of association rules - support, confidence and lift
          on the dataset. First, I will find the association rules on the 
          data and calculate their confidence. Then I will save for each feature
           the maximum value of the confidence from all of the discovered rules. The 
           last step would be re-ordering the features by the max values they got. In this
            way I can determine which of them are most informative and valuable. """)

        st.write("""### Experiment Plan""")
        st.markdown("""In order to test the effectiveness of my
        solution, I will conduct a comparison experiment on the same 
        databases by running the ReliefF algorithm, which is an algorithm that
         returns the best subset of features of the wanted size. The ReliefF 
         works different than my solution, but has the same purpose. After doing the 
         experiment (and run my algorithm on the datasets), I will compare my output 
         with the ReliefF’s one.""")
