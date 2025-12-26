import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Feature engineering - Create binary target (Pass/Fail based on average score)
df['final_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['pass_fail'] = (df['final_score'] >= 70).astype(int)  # 1 for pass, 0 for fail

df['parental level of education'] = df['parental level of education'].astype('category').cat.codes
df['lunch'] = df['lunch'].astype('category').cat.codes
df['test preparation course'] = df['test preparation course'].astype('category').cat.codes

# Features and target
X = df[['math score', 'reading score', 'writing score',
        'parental level of education', 'lunch', 'test preparation course']]
y = df['pass_fail']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "pass_fail_model.pkl")
print("Model trained and saved successfully!")
