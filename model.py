import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. Exploratory Data Analysis Plots
plt.figure()
train['Survived'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.xticks([0, 1], ['Did Not Survive', 'Survived'], rotation=0)
plt.title('Survival Count')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.savefig('graph1_survival_count.png')
plt.close()

plt.figure()
sns.countplot(data=train, x='Sex', hue='Survived')
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Did Not Survive', 'Survived'])
plt.tight_layout()
plt.savefig('graph2_gender_survival.png')
plt.close()

plt.figure()
sns.countplot(data=train, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(['Did Not Survive', 'Survived'])
plt.tight_layout()
plt.savefig('graph3_class_survival.png')
plt.close()

# 3. Data Preprocessing (Train)
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train = train.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'])
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)

X = train.drop('Survived', axis=1)
y = train['Survived']

# 4. Train/Validation Split & Feature Scaling
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 5. Model Training & Evaluation
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# 6. Test Data Preprocessing & Predictions
test_ids = test['PassengerId']
test['Age'] = test['Age'].fillna(train['Age'].median())
test['Fare'] = test['Fare'].fillna(train['Fare'].median())
test = test.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'])
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)
test = test.reindex(columns=X.columns, fill_value=0)

test_scaled = scaler.transform(test)
predictions = model.predict(test_scaled)

submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
submission.to_csv('submission.csv', index=False)

print("Total predicted:", len(predictions))
print("Survived:", predictions.sum())
print("Did not survive:", len(predictions) - predictions.sum())
