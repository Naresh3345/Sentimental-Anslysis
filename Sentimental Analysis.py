import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle


nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv(r"D:\\Year-3 Term-1 Internship\\Sentimental Analysis\\sentiment_tweets3.csv")


df.rename(columns={"message to examine":"text", "label (depression result)":"label"}, inplace=True)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()  
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  
    return text

df['text'] = df['text'].apply(preprocess_text)


min_word_count = 3
df = df[df['text'].apply(lambda x: len(x.split()) >= min_word_count)]


print("Class distribution before balancing:")
print(df['label'].value_counts())


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

model = MultinomialNB()
model.fit(X_train_tfidf_resampled, y_train_resampled)


with open('naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([preprocessed_text]) 
    prediction = model.predict(text_tfidf)
    return prediction[0]

example_texts = [
    "Today has been one of the best days in recent memory! I got promoted at work, and my colleagues threw a surprise party for me. I can't believe how lucky I am to have such amazing people around me. It's truly been a day full of joy and celebration",
    "I can't believe this is happening. So disappointed with everything right now",
    "I am neither happy nor sad, just feeling okay."
]

for text in example_texts:
    prediction = predict_sentiment(text)
    sentiment = "sad" if prediction == 1 else "happy"
    print(f"Input: {text}")
    print(f"Predicted Sentiment: {sentiment}\n")