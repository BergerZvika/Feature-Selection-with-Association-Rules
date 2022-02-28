import streamlit as st

from Pages.page import Page
from config import Config


class DatasetsPage(Page):
    def show_page(self):
            st.write("""# Dataset Information""")
            st.write("In order to find the importance of features in a dataset, "
                     "we've experimented a number of different datasets. Each dataset contains a large amount of "
                     "records and a wide variety of features.")
            st.write("You can see the datasets below:")

            st.write("""#### Dataset""")
            data = st.selectbox("Choose Dataset:",
                                   Config.datasets_names)

            dataset = Config.datasets[data]

            st.dataframe(dataset)
            st.write("""#### More information about our dataset""")
            st.write("Here you can find statistical information about each feature of the data:")
            st.write(dataset.describe())
