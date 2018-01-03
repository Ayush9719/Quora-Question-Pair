# Quora-Question-Pair

This is a Kaggle compition from Quora to find the question pairs having the same intent using machine learning and Natural Language Processing.

The competition's link is [here.](https://www.kaggle.com/c/quora-question-pairs)

In the description of this compitition, quora has mentioned that they have been using **Random Forest model** to identify duplicate questions and they are asking the kagglers to apply advanced deep learning techniques. But we tackled this problem to understand the concepts of NLP and basic Machine Learning Models.

### In this project there are two scripts:
#### 1. Feature Engineering
Initially we attained the difference between the length of questions (with and without spaces) as basic features. Then we used fuzzywuzzy package to extract fuzzy ratios for every question pair. The documentation is available [here.](https://pypi.python.org/pypi/fuzzywuzzy) We have also used the Google News Vectors to implement Word2Vec and extract distances like:
* Euclidean
* Cosine
* CityBlock
* Jaccard
* Minkowski
* Canberra
* Braycurtis

#### 2. Classification
For classification, we have used the following machine learning models:
  * Nearest Neighbors
  * XGBoost
  * Decision Tree
  * Random Forest
  * ExtraTreesClassifier
  * AdaBoost
  
  This project helped us to understand concepts of NLP and basic Machine Learning models.
