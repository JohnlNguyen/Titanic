import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data_dir = 'Input/'
train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')
full = pd.concat([train, test], axis=0)

# Feature Engineering

full['Sex'] = np.where(full['Sex'] == 'male', 1, 0)  # male is 1, female is 0
full['First_class'] = np.where(full['Cabin'].isnull(), 1, 0)  # 1 if 1st class, 0 if not
full = full.drop(['Ticket', 'Cabin'], axis=1)


def transform_age(row):
    """ Age statistics: 1st Q = 20 median = 28 3rd Q = 38 max = 80
    :param row: 
    :return: group the age by the above age range
    """
    if row['Age'] >= 38:
        val = 'Old'
    elif row['Age'] >= 20 and row['Age'] < 38:
        val = 'Young'
    else:
        val = 'Child'
    return val


def discretize_family(row):
    """Transform family size into 3 categories 
    :param row: 
    :return: family group such as single (travels alone)
    small family: 2 to 4 members
    large family: 5 or more members
    """
    if row['Fsize'] == 1:
        val = 'Single'
    if row['Fsize'] >= 2 and row['Fsize'] <= 4:
        val = 'Small'
    else:
        val = 'Large'
    return val


def transform_embarked(row):
    """Transform Embarked size into 4 int values 
        :param row: 
        :return: 
        """
    if row['Embarked'] == 'S':
        val = 1
    elif row['Embarked'] == 'C':
        val = 2
    elif row['Embarked'] == 'Q':
        val = 3
    else:
        val = 0
    return val


full['Age_group'] = full.apply(transform_age, axis=1)
full['Age_group'] = le.fit_transform(full['Age_group'])
full['Fsize'] = full['SibSp'] + full['Parch'] + 1
full['Family'] = full.apply(discretize_family, axis=1)
full['Family'] = le.fit_transform(full['Family'])
full['Embarked'] = full.apply(transform_embarked, axis=1)
full['Title'] = np.array(full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False))
full['Title'] = full['Title'].replace(['Ms', 'Mme', 'Mlle'], 'Miss')
full['Title'] = full['Title'].replace(['Lady', 'Sir', 'Major', 'Capt', 'Master'], 'Rich')
full['Title'] = full['Title'].replace(['Col', 'Capt', 'Countess', 'Jonkheer',
                                       'Dona', 'Don', 'Rev', 'Dr', ], 'Rare')
# Check out the correlation between Title and Survival Rate
# full[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
full['Title'] = le.fit_transform(full['Title'])
full = full.drop(['SibSp', 'Parch', 'Name'], axis=1)

# Getting the age for NA values
full['Age'] = full['Age'].fillna(full['Age'].median())
full['Fare'] = full['Fare'].fillna(0)

train_df = full.iloc[0:891, :]
test_df = full.iloc[891:1309, :].drop('Survived', axis=1)

Y_train = train_df["Survived"].astype(int)
X_train = train_df.drop(["Survived", "PassengerId"], axis=1)
X_test = test_df.drop("PassengerId", axis=1).copy()

# Prediction

# Logistic Regression
# logmodel = LogisticRegression()
# logmodel.fit(X_train, Y_train)
# predictions = logmodel.predict(X_test)
# validation = logmodel.predict(X_train)
# print(classification_report(validation, Y_train))  # 83%

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, Y_train)
rfc_predict = rfc.predict(X_test)
rfc_validate = rfc.predict(X_train)
print(classification_report(rfc_validate, Y_train))  # 99%

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": rfc_predict
})
submission.to_csv('output/submission.csv', index=False)
