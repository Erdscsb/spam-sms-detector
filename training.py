import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("C:/Users/User/Downloads/Egyetem/Beadando/spam.csv",encoding='ISO-8859-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])  # Convert text to numerical data
y = df['label']  # Target labels (0 or 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

def predict_spam(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the trained model
with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

# Example
print(predict_spam("Congratulations! You've won a free ticket to Bali. Call now!"))
print(predict_spam("Don't forget about dinner at 7 PM. See you then!"))
print(predict_spam("Free entry in a weekly competition to win a brand new car! Text WIN to 12345."))
print(predict_spam("You are selected for a free vacation package. Call 1-800-NOW to book!"))


