📰 Fake News Detection System

📌 Overview
This project is a Machine Learning-based web application that detects whether a news article is Fake or Real using Natural Language Processing (NLP) techniques.

🚀 Features
Classifies news articles as Fake or Real
Uses TF-IDF for text vectorization
High accuracy (~98.5%)
Interactive web interface using Streamlit

🛠️ Tech Stack
Python
Pandas
Scikit-learn
NLP (Text Processing)
Streamlit
Joblib

📂 Dataset
Fake.csv
True.csv

⚙️ How It Works
Data preprocessing (cleaning text, removing noise)
Feature extraction using TF-IDF
Model training using Logistic Regression
Prediction on user input via web app

📊 Model Performance
Accuracy: 98.5%
Precision, Recall, F1-score evaluated

💻 Installation & Setup
1. Clone the repository
  git clone https://github.com/your-username/fake-news-detection.git
  cd fake-news-detection
2. Install dependencies
  pip install -r requirements.txt
3. Run the app
  streamlit run app.py

📸 Usage
Enter a news article in the text box
Click Check News
Get prediction: Fake or Real

📁 Project Structure
├── Fake.csv
├── True.csv
├── app.py
├── model (lr.jb)
├── vectorizer.jb
└── README.md

🔮 Future Improvements
Add deep learning models (LSTM, BERT)
Improve UI design
Deploy on cloud (AWS/Heroku)

👨‍💻 Author:
ABDUL WARIS
