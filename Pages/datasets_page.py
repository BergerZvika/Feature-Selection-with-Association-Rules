import streamlit as st

from Pages.page import Page
from config import Config


class DatasetsPage(Page):
    def show_page(self):
            st.write("""# Dataset Information""")

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

            st.dataframe(dataset)
            st.write("""#### More information about our dataset""")
            st.write(dataset.describe())
