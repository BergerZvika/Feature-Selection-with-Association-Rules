# Feature Selection with Association Rules
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
The site contains 7 pages. The first 6 of them depend on each other. In order to continue to the next page, you need to pass the previous page first:
- Introduction page - an intro about our project, our goal and the ways to get there.
- Datasets page - shows in a table a dataset from all of our datasets that the system works on, including a table with other information such as mean and max value of every feature. You can choose to see every dataset from our veraity.
- Association rules page - finds all of the association rules in 80% of shuffeled dataset - which is the part of the dataset that is taken for training the model. In the page, you choose a dataset, a support and confidence thresholds and click the "find association rules" button. A table with all rules found will be shown with their support, confidence, lift and other values.
- Association rules analysis page - this page first shows all of the rules that were found in the last page. After that, you choose one feature from all of the dataset's features and a table with all of the rules that include that feature will be shown.
- Feature selection page - in this page you choose the number of top features you want to detect in every concept (support, confidence, lift, distance to 1 of lift). A table with the top K features of each concept will be shown.
- Model comparison page - you choose a model to predict the features from a wide veraity of models options. After clicking the "predict" button, the results will be shown on the screen with graphs that compares the predictions againts the true value. The model you chose is trained by the 80% data that's been selected in the association rules page, and by them it predicts and compare to the test's real values.
- Test page - this page does not depend on the previous pages. Here you can do multiple tests on the datasets and perform in one click the whole website goal. First choose a dataset you want to test, then select the feature from all of the dataset's features and then choose the amount of tests you want. After clicking "Test", it will start and you will see on the screen the parts of the test that are done. Every part of the test includes different values of support threshold, confidence threshold and K (to find top-K). The test includes: 1. Finding association rules 2. Analysing the rules 3. Performing feature selection 4. Comparing the models.
After the test ends, a folder with all of the information will be opened. This folder includes a statistics page, the test and train data and all of the information that is given in the other pages (graphs of each model afor different K's, a csv file with all of the association rules that found, a csv file with the analysis of the rules and a csv file of the models comparisons).
More information about this page is written in the page.
