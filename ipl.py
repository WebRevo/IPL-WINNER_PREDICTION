import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Read the CSV file
df = pd.read_csv('matches.csv')

# Filter columns
new_df = df[['team1', 'team2', 'winner', 'toss_decision', 'toss_winner']]

# Drop rows with missing values
new_df.dropna(inplace=True)

# Encode team names using LabelEncoder
teams_encoder = LabelEncoder()
new_df['team1'] = teams_encoder.fit_transform(new_df['team1'])
new_df['team2'] = teams_encoder.transform(new_df['team2'])
new_df['toss_winner'] = teams_encoder.transform(new_df['toss_winner'])

# Encode toss_decision using a dictionary
toss_decision_mapping = {'field': 0, 'bat': 1}
new_df['toss_decision'] = new_df['toss_decision'].map(toss_decision_mapping)

# Encode the target variable (winner)
new_df['winner'] = teams_encoder.transform(new_df['winner'])

# Split the data into features (X) and target (y)
X = new_df[['team1', 'team2', 'toss_decision', 'toss_winner']]
y = new_df['winner']

# Balance the classes (optional)
# You can add your class balancing code here if needed.

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Train SVM model
model1 = SVC().fit(X_train, y_train)
score1 = model1.score(X_test, y_test)

# Train Decision Tree model
model2 = DecisionTreeClassifier().fit(X_train, y_train)
score2 = model2.score(X_test, y_test)

# Train Random Forest model
model3 = RandomForestClassifier(n_estimators=250).fit(X_train, y_train)
score3 = model3.score(X_test, y_test)

# Save the Random Forest model to a file
with open('model.pkl', 'wb') as f:
    pkl.dump(model3, f)

# Load the Random Forest model from the file (optional)
# with open('model.pkl', 'rb') as f:
#     model = pkl.load(f)

# Test predictions
test_data = np.array([2, 4, 1, 4]).reshape(1, -1)
prediction1 = model1.predict(test_data)
prediction2 = model2.predict(test_data)
prediction3 = model3.predict(test_data)

print("SVM Model Prediction:", prediction1)
print("Decision Tree Model Prediction:", prediction2)
print("Random Forest Model Prediction:", prediction3)

# Print model scores
print("SVM Model Score:", score1)
print("Decision Tree Model Score:", score2)
print("Random Forest Model Score:", score3)
