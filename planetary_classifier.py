# Install required packages
# Run these commands in your terminal:
# pip install scikit-learn pandas numpy nltk matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')

# Load planetary dataset (replace with your actual file)
df = pd.read_csv(r"C:\Users\pratyaksh gupta\Downloads\Planetary classifier project\planetary_dataset.csv")

print("Available columns:", df.columns.tolist())
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass Distribution:")
# print(df['planet_type'].value_counts())
print("\nDataset Info:")
print(df.info())

def preprocess_text(text):
    """
    Comprehensive text preprocessing pipeline
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Rejoin tokens
    return ' '.join(tokens)

# Drop rows where target 'Prediction' is missing
df = df.dropna(subset=['Prediction'])

# Now prepare features and target
X = df.drop('Prediction', axis=1)
y = df['Prediction'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Classes in training set: {y_train.value_counts().to_dict()}")

# Create TF-IDF vectorizer with carefully chosen parameters
vectorizer = TfidfVectorizer(
    max_features=1000,        # Limit vocabulary size
    ngram_range=(1, 2),       # Use unigrams and bigrams
    min_df=2,                 # Ignore terms appearing in fewer than 2 documents
    max_df=0.8,               # Ignore terms appearing in more than 80% of documents
    sublinear_tf=True,        # Apply sublinear tf scaling
    stop_words='english'      # Additional stopword removal
)

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"Feature names sample: {vectorizer.get_feature_names_out()[:10]}")

# Handle missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Prediction')
cat_cols = df.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

df = df.dropna(subset=['Prediction'])  # Drop rows missing target

# Encode categorical columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Prepare features accordingly (drop 'Prediction' column)
X = df.drop('Prediction', axis=1)
y = df['Prediction'].astype(int)

# Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize, train and evaluate classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, clf in classifiers.items():
    print(f"\n=== {name} ===")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Random Forest params:", grid_search.best_params_)
print("Best cross-val accuracy:", grid_search.best_score_)

import matplotlib.pyplot as plt
import seaborn as sns

best_rf = grid_search.best_estimator_

feat_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Feature Importances from Random Forest")
plt.show()

joblib.dump(best_rf, 'best_random_forest_model.pkl')

# Example new planetary data -- values should be preprocessed same way as training data
new_sample = {
    'Atmospheric Density': 0.5,
    'Surface Temperature': -1.2,
    'Gravity': 0.75,
    'Water Content': 0.3,
    'Mineral Abundance': -0.2,
    'Orbital Period': 1.1,
    'Proximity to Star': 0.8,
    'Magnetic Field Strength': 3,   # encoded categorical value
    'Radiation Levels': 4,          # encoded categorical value
    'Atmospheric Composition Index': 0.7
}

# Convert to DataFrame with a single row
new_sample_df = pd.DataFrame([new_sample])

# Assuming you've saved the best model as 'best_random_forest_model.pkl'
model = joblib.load('best_random_forest_model.pkl')

# Predict the class
predicted_class = model.predict(new_sample_df)[0]

# For classifiers that support predict_proba (like Random Forest), get prediction probabilities
if hasattr(model, "predict_proba"):
    confidence = model.predict_proba(new_sample_df).max()
else:
    confidence = None

print(f"Predicted class: {predicted_class}")
if confidence is not None:
    print(f"Confidence score: {confidence:.3f}")