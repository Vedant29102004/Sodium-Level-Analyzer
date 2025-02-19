import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Sample dataset
data = {
    'age': [25, 35, 45, 55, 65],
    'blood_pressure': [120, 130, 140, 150, 160],
    'diet': ['healthy', 'unhealthy', 'healthy', 'unhealthy', 'healthy'],
    'sodium_level': ['average', 'high', 'low', 'high', 'average']
}

# Convert to DataFrame
data = pd.DataFrame(data)

# Encode categorical features
encoder_diet = LabelEncoder()
data['diet_encoded'] = encoder_diet.fit_transform(data['diet'])

encoder_sodium = LabelEncoder()
data['sodium_level_encoded'] = encoder_sodium.fit_transform(data['sodium_level'])

# Define features (X) and target (y)
X = data[['age', 'blood_pressure', 'diet_encoded']]
y = data['sodium_level_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model and encoder
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump((model, encoder_sodium), file)

print("Model saved successfully.")
