from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('naive_bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    return text.lower()  # Simple preprocessing, you can add more steps as needed

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    preprocessed_text = preprocess_text(input_text)
    text_tfidf = vectorizer.transform([preprocessed_text]) 
    prediction = model.predict(text_tfidf)
    
    sentiment = "Sad" if prediction[0] == 1 else "happy"
    
    return render_template('index.html', input_text=input_text, sentiment=sentiment)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
