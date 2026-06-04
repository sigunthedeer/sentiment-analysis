# Movie Review Sentiment Analysis (NLP)

> Can a model read a movie review and tell whether it's positive or negative? 
> Using 50,000 IMDB reviews, this project builds an NLP sentiment classifier - 
> from raw, messy text to a model that reads tone with 89.5% accuracy.

## The Question

Given the free-text of a movie review, can we predict its sentiment - and which 
words most strongly signal a positive or negative opinion?

## Key Results

- **89.5% accuracy** and **0.962 ROC-AUC** on 10,000 held-out reviews
- Balanced performance across both positive and negative classes

## Key Findings

- Strongest positive signals: "great", "excellent", "perfect", "amazing"
- Strongest negative signals: "worst", "awful", "bad", "boring"
- **Negative words carried higher weights than positive ones** - suggesting 
  unhappy reviewers use more emphatic, decisive language
- Two-word phrases like "the best" and "the worst" were captured via bigram 
  features, preserving context that single words miss

## The NLP Pipeline

This project demonstrates the full text-classification workflow:
1. **Text cleaning** with regular expressions - removing HTML tags, lowercasing, stripping punctuation
2. **TF-IDF vectorization** with unigrams and bigrams - turning text into a numeric matrix
3. **Logistic Regression** - handling high-dimensional sparse text data
4. **Model interpretation** - extracting which words drive predictions

## Tools Used

Python · pandas · NumPy · scikit-learn · matplotlib

## Dataset

IMDB 50K Movie Reviews (Kaggle). Note: the CSV (~66 MB) is excluded from this 
repo via .gitignore - download it from Kaggle and place it in the project folder to run.
