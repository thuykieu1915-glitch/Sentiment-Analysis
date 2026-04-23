# =============================
# SENTIMENT ANALYSIS PROJECT
# =============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. DATA (simple dataset)
data = {
    'text': [
        'I love this product',
        'This is amazing',
        'Very good experience',
        'I am happy with the service',
        'Absolutely fantastic',

        'I hate this',
        'Very bad experience',
        'This is terrible',
        'I am disappointed',
        'Worst product ever'
    ],
    'label': [
        1, 1, 1, 1, 1,   # positive
        0, 0, 0, 0, 0    # negative
    ]
}

df = pd.DataFrame(data)

print("=== DATA ===")
print(df)

# 2. FEATURES & LABEL
X = df['text']
y = df['label']

# 3. VECTORIZE TEXT
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. TRAIN MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. PREDICT
y_pred = model.predict(X_test)

# 7. EVALUATE
print("\n=== RESULT ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:")
print(classification_report(y_test, y_pred))

# 8. TEST CUSTOM INPUT
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)[0]
    return "POSITIVE" if result == 1 else "NEGATIVE"

print("\n=== CUSTOM TEST ===")
print("I love this phone:", predict_sentiment("I love this phone"))
print("This is so bad:", predict_sentiment("This is so bad"))

# 9. SAVE MODEL
import joblib
joblib.dump(model, "sentiment_model.pkl")

print("\nModel saved!")