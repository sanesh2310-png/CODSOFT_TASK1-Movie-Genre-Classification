# CODSOFT_TASK1 — Movie Genre Classification

Predicts a movie's genre from its plot summary using TF-IDF and classic machine learning classifiers.

##📌 Problem Statement

Given a movie's plot summary, predict its genre from 27 possible categories (drama, comedy, documentary, horror, thriller, etc.), using only the text of the plot.

## 📂 Dataset

[Genre Classification Dataset - IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) (Kaggle)

- `train_data.txt` — 54,214 labeled examples (format: `ID ::: TITLE ::: GENRE ::: PLOT`)
- `test_data_solution.txt` — 54,200 labeled examples used for evaluation
- `test_data.txt` — 54,200 unlabeled examples used for final blind predictions

## 🛠️ Approach

1. **Data Cleaning**
   - Lowercased all plot text
   - Removed punctuation and numbers
   - Collapsed extra whitespace
   - Stripped stray spaces from genre labels

2. **Feature Extraction**
   - TF-IDF vectorization (unigrams + bigrams)
   - Top 20,000 features
   - English stopwords removed

3. **Model Training**
   Trained and compared three classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM

4. **Evaluation**
   Compared accuracy on the labeled test set (`test_data_solution.txt`), then used the best model to predict genres for the blind test set (`test_data.txt`).

## 📊 Results

| Model | Accuracy |
|---|---|
| Naive Bayes | 51.0% |
| Logistic Regression | 59.2% ✅ (best) |
| Linear SVM | 57.8% |

Logistic Regression performed best overall. It's strongest on frequent, distinct genres like `western` (F1: 0.81), `documentary` (F1: 0.75), and `drama` (F1: 0.64). It struggles with rare or overlapping genres such as `biography`, `history`, and `war` — largely due to class imbalance in the dataset.

## 📁 Repository Contents

- `CODSOFT_TASK1_Movie_Genre_Classification.ipynb` — full notebook (data cleaning, TF-IDF, model training, evaluation, predictions)
- `predictions.csv` — predicted genres for the blind test set
- `README.md` — this file

## ▶️ How to Run

1. Open the notebook in Google Colab or Jupyter
2. Upload `train_data.txt`, `test_data.txt`, and `test_data_solution.txt`
3. Run all cells in order

## 🔮 Possible Improvements

- Use word embeddings (Word2Vec, GloVe) or transformer-based models instead of TF-IDF
- Address class imbalance with oversampling or class weighting
- Hyperparameter tuning via grid search
