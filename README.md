﻿# Feature Selection with Association Rules
[![built with Streamlit](https://img.shields.io/badge/built%20with%20-Streamlit-brightgreen)](https://www.streamlit.io/)


Tabular data science final project. A Python project that finds association rules on datasets, analyzes them, predicts features and calculates them.

## Authors
Zvi Berger

Ofir Nassimi

## Run
- Download the project
- Install all requiments
```python
 pip install -r requirements.txt
```
- Run the server
 ```'python
 streamlit run main.py
```
- Copy the url to open browser

## Site Review
The site contains 7 pages:
- Introduction page - an intro about our project, our goal and the ways to get there.
- Datasets page - shows in a table a dataset from all of our datasets that the system works on, including a table with other information such as mean and max value of every feature. You can choose to see every dataset from our veraity.
- Association rules page - finds all of the association rules in the dataset. In the page, you choose a dataset, a support and confidence thresholds and click the "find association rules" button. A table with all rules found will be shown with their support, confidence, lift and other values.
- Association rules analysis page - this page first shows all of the rules that were found in the last page. After that, you choose one feature from all of the dataset's features and a table with all of the rules that include that feature will be shown.
- Feature selection page - in this page you choose the number of top features you want to detect in every concept (support, confidence, lift, distance to 1 of lift). A table with the top K features of each concept will be shown.
- Model comparison page - you choose a model to predict the features by from a wide veraity of models options. After clicking the "predict" button, the results will be shown on the screen with graphs that compares the predictions againts the true value.
