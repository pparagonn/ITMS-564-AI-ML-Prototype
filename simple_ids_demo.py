import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)  
n_samples = 1000  

data = {
    'Destination Port': np.random.randint(0, 65535, n_samples),
    'Flow Duration': np.random.randint(1, 120000, n_samples),
    'Total Fwd Packets': np.random.randint(1, 20, n_samples),
    'Total Backward Packets': np.random.randint(1, 20, n_samples),
    'Total Length of Fwd Packets': np.random.randint(1, 10000, n_samples),
    'Total Length of Bwd Packets': np.random.randint(1, 10000, n_samples),
    'Fwd Packet Length Max': np.random.randint(1, 1500, n_samples),
    'Fwd Packet Length Min': np.random.randint(0, 1500, n_samples),
    'Fwd Packet Length Mean': np.random.random(n_samples) * 1500,
    'Fwd Packet Length Std': np.random.random(n_samples) * 500,
    # You can simulate more features here...
    'Label': np.random.choice(['BENIGN', 'DDoS'], n_samples)
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=le.classes_))
