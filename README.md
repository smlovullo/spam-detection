# Spam Detection

## Description

This project explores a text dataset of SMS messages labeled as spam or ham with the goal of creating a binary text classifier that will be able to determine whether a body of text is a phishing, scam, or spam message, or if it is not.  

## Citation

The following dataset was used to develop this model:  

Almeida,Tiago and Hidalgo,Jos. (2012). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.  

## Work

The jupyter notebook files in the notebooks folder contain the demonstratable work for this project. In the file `eda.ipynb`, the initial exploration of the dataset takes place. The file `modeling.ipynb` runs all of the modeling experiments to figure out which model should be selected for spam classification. Finally, the `train.ipynb` file runs the code to create a new model, and the `classify.ipynb` file uses the model created from `train.ipynb` to classify text as "ham" or "spam".  

I started this project expecting to start with taking a "bag of words" approach, since the raw text data alone can't be used to train any models, but found that other countable features about the text held more predictive power for this classification than I anticipated. I spent time creating sets of models for each of these approaches (as well as a combined approach) to training a model, and comparing results to determine which option was best. Ultimately, the "bag of words" approach still showed the best results for creating a model using more "traditional" statisitcal modeling methods.

Future plans for this project would be to create a deployable model utilizing a Python REST API and hosted in the cloud, and to use it on a Discord server as a bot to detect scams.
