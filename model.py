import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data_dir = 'Input/'
train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')
full = pd.concat([train, test], axis=0)

# Feature Engineering
def split_ticket(row):
    row_arr = row["Ticket"].split(" ")
    if len(row_arr) > 1:
        return row_arr[1]
    else:
        return row_arr[0]

full['Sex'] = np.where(full['Sex'] == 'male', 1, 0)  # male is 1, female is 0
full['inCabin'] = np.where(full['Cabin'].isnull(), 1, 0)  # 1 if 1st class, 0 if not
full['Ticket'] = full.apply(split_ticket, axis=1)
full['Ticket'] = le.fit_transform(full['Ticket'])
full = full.drop(['Cabin'], axis=1)


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
        val = 'Alone'
    elif row['Fsize'] == 2:
        val = 'Couple'
    elif row['Fsize'] > 2 and row['Fsize'] <= 4:
        val = 'Small_Fam'
    elif row['Fsize'] > 4 and row['Fsize'] <= 8:
        val = 'Medium_Fam'
    else:
        val = 'Large_Fam'
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


# Adding has a is mother Feature
def is_mother(row):
    """A Child who has a mother will simply be someone under 18 years of age and has a mother who 
    1) female
    2) is over 18
    3) Parch > 0
    4) does not have the title ‘Miss’
    :param row: 
    :return: 0 if does not have mom
    1 if does 
    """
    if row['Sex'] == 0 and row['Parch'] > 0 and row['Title'] != 'Miss':
        val = 1
    else:
        val = 0
    return val


def transform_age_size(row):
    if row['Age_group*Fsize'] < 2:
        val = 0
    if row['Age_group*Fsize'] >= 2 and row['Age_group*Fsize'] < 4:
        val = 1
    else:
        val = 3
    return val


full['Age_group'] = full.apply(transform_age, axis=1)
full['Age_group'] = le.fit_transform(full['Age_group'])
full['Fsize'] = full['SibSp'] + full['Parch'] + 1
full['Age_group*Fsize'] = full['Age_group'] * full['Fsize']
# full['Age_group*Fsize'] = full.apply(transform_age_size, axis=1)
full['Family'] = full.apply(discretize_family, axis=1)
full['Family'] = le.fit_transform(full['Family'])
full['Embarked'] = full.apply(transform_embarked, axis=1)
full['Title'] = np.array(full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False))

full['Title'] = full['Title'].replace(['Ms', 'Mme', 'Mlle'], 'Miss')
full['Title'] = full['Title'].replace(["Don", "Rev", "Major", "Sir", "Col", "Capt", "Jonkheer"], "Mr")
full['Title'] = full['Title'].replace(['Master'], 'Child')
full['Title'] = full['Title'].replace(['Lady'], 'Mrs')
full['Title'] = full['Title'].replace(['Countess', 'Dona', 'Don', 'Rev', 'Dr'], 'Rare')
full['isMother'] = full.apply(is_mother, axis=1)

# Check out the correlation between Title and Survival Rate
# full[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
full['Title'] = le.fit_transform(full['Title'])
full['isAlone'] = np.where(full['Fsize'] == 1, 1, 0)
full = full.drop(['SibSp', 'Parch', 'Name'], axis=1)

# Getting the age for NA values
# full['Age'] = full['Age'].fillna(full['Age'].mean())

full['Fare'] = full['Fare'].fillna(0)

train_df = full.iloc[0:891, :]
test_df = full.iloc[891:1309, :].drop('Survived', axis=1)

x_train_age = train_df[train_df['Age'].notnull()].drop('Age', axis=1)
y_train_age = train_df[train_df['Age'].notnull()]['Age'].astype(int)
test_age = train_df[train_df['Age'].isnull()].drop('Age', axis=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_age, y_train_age)
pred = knn.predict(test_age)

Y_train = train_df["Survived"].astype(int)
X_train = train_df.drop(["Survived", "PassengerId"], axis=1)

X_test = test_df.drop("PassengerId", axis=1).copy()
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())

# Prediction

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, Y_train)
rfc_predict = rfc.predict(X_test)
rfc_validate = rfc.predict(X_train)
acc = round(rfc.score(X_train, Y_train) * 100, 2)
feat_labels = X_train.keys()
print(classification_report(rfc_validate, Y_train))  # 99%
print(acc)

for feature in sorted(zip(feat_labels, rfc.feature_importances_)):
    print(feature)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": rfc_predict
})

sample = pd.read_csv(data_dir + 'gender_submission.csv')
mis = sample[submission['Survived'] != sample['Survived']]

mis_info = pd.merge(mis, full, on=['PassengerId'])
print(len(mis_info) / len(sample))

if len(mis_info) / len(sample) < .14:
    submission.to_csv('output/submission.csv', index=False)
