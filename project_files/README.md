# Fake News Detection using Machine Learning

A comprehensive fake news detection system using multiple datasets and ML algorithms.

## 🎯 Project Overview

This project implements fake news detection using:
- 3 different datasets (combined 159,000+ samples)
- Multiple ML algorithms (XGBoost, Logistic Regression, MLP)
- Comprehensive preprocessing and feature engineering
- TF-IDF vectorization with n-grams

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| Logistic Regression | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| MLP | XX.XX% | XX.XX% | XX.XX% | XX.XX% |

*(Update with your actual results)*

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.7+
pip install -r requirements.txt
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt
```

### Usage
```python
import joblib

# Load model and vectorizer
model = joblib.load('models/xgboost_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Predict
text = "Your news article text here"
text_vectorized = vectorizer.transform([text])
prediction = model.predict(text_vectorized)

print("Fake" if prediction[0] == 0 else "Real")
```

## 📁 Project Structure

```
fake_news_detection/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Preprocessed data
├── models/               # Trained models (.pkl files)
├── visualizations/       # Performance charts
├── results/              # Model results (JSON/CSV)
├── README.md
└── requirements.txt
```

## 📈 Datasets Used

1. **Figshare Fake-True News Dataset** (76 MB)
2. **Kaggle Fake-Real News Dataset** (114 MB)  
3. **WelFake Dataset** (233 MB)

## 🛠️ Technologies

- **Python 3.x**
- **scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **pandas** - Data manipulation
- **NLTK** - Text preprocessing
- **matplotlib/seaborn** - Visualization

## 📊 Features

- Text cleaning and preprocessing
- Stopword removal
- TF-IDF vectorization with bigrams
- Multiple model comparison
- Balanced dataset handling
- Comprehensive evaluation metrics

## 🔍 Model Details

### XGBoost
- Max depth: 6
- Learning rate: 0.1
- N estimators: 100

### Logistic Regression
- Max iterations: 1000
- Multi-core processing

### MLP (Neural Network)
- Hidden layers: (100, 50)
- Activation: ReLU
- Early stopping enabled

## 📊 Visualizations

See `visualizations/` folder for:
- Confusion matrices for all models
- Performance comparison charts
- Dataset distribution plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

Your Name

## 🙏 Acknowledgments

- Kaggle for dataset hosting
- Figshare for dataset access
- scikit-learn community
- XGBoost developers
