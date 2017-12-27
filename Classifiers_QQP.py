import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier 
from sklearn import preprocessing

names = ["Nearest Neighbors", "XGBoost",
         "Decision Tree", "Random Forest","ExtraTreesClassifier", "AdaBoost"]

classifiers = [
    KNeighborsClassifier(10),
    xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100,max_features=7),
    ExtraTreesClassifier(max_depth=5,n_estimators=100,max_features=7),
    AdaBoostClassifier(base_estimator='DecisionTreeClassifier', learning_rate=0.03,n_estimators=10)]

df = pd.read_csv('/Quora Question Pairs/Codes_Data/quora_features.csv')
import pandas as pd

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

Z= df.drop(['question1','question2'], 1)
print Z.head()
Z=clean_dataset(Z)
y = Z['is_duplicate']
X = Z.drop(['is_duplicate'], 1)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, stratify=Z.is_duplicate,random_state=42)
feat_test=DecisionTreeClassifier(min_samples_split=0.1)
feat_test.fit(X_train, y_train)
print "\n\nImportant Features:\n",feat_test.feature_importances_

# iterate over classifiers
for name, clf in zip(names, classifiers)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print name+' Score:',score
        prediction1 = clf.predict(X_test)
        np.set_printoptions(precision=2)