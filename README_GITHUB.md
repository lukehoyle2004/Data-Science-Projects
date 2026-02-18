# Data Science Projects (Python)

This repo contains two small projects written in Python:

## 1) Email Spam Classifier (Naive Bayes)
- Script: `email_spam_classifier.py`
- Techniques: text preprocessing, Bag-of-Words vectorization, Multinomial Naive Bayes
- Metrics: accuracy, precision, recall, F1, confusion matrix
- Optional Laplace smoothing (alpha) tuning

Run:
```bash
pip install -r requirements.txt
python email_spam_classifier.py --csv path/to/enronSpamSubset.csv --text_col Body --label_col Label --tune_alpha
```

## 2) Stock Price Prediction (Linear Regression)
- Script: `stock_price_prediction.py`
- Features: SMA_10 and SMA_30
- Model: Linear Regression
- Metrics: MSE, RMSE, R^2
- Uses Yahoo Finance data via `yfinance`

Run:
```bash
pip install -r requirements.txt
python stock_price_prediction.py --ticker LMT --start 2016-01-01 --end 2026-01-22 --plot
```

## Notes
- Time series split uses `shuffle=False` to avoid data leakage.
- For the spam project, NLTK resources will be downloaded automatically if needed.
