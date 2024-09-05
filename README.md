# Movie Reviews Sentiment Classification

**Author**: Ramya Chowdary Patchala  
**Email**: rpatchal@syr.edu  

## Project Overview

This project performs sentiment classification on the Kaggle Movie Reviews Dataset using Natural Language Processing (NLP) techniques. The goal is to classify movie reviews into positive, negative, and neutral sentiments. The best model, based on a Naïve Bayes algorithm with a comprehensive feature function, achieved an accuracy (F-1 score) of 85.1% using 10-fold cross-validation.

## Dataset

- **Source**: [Kaggle Movie Reviews Dataset](https://www.kaggle.com/datasets)
- **Training Data**: 156,060 phrases with corresponding sentiment labels
- **Test Data**: 66,292 phrases
- **Sentiment Labels**: 
  - 0: Negative
  - 1: Slightly Negative
  - 2: Neutral
  - 3: Slightly Positive
  - 4: Positive

For classification purposes, sentiments were simplified as follows:
- **Negative**: 0, 1
- **Neutral**: 2
- **Positive**: 3, 4

## Methods

### Text Preprocessing
1. **Tokenization**: Split reviews into individual tokens.
2. **Normalization**: Converted all tokens to lowercase.
3. **Lemmatization**: Reduced words to their base forms.
4. **Filtering**: Removed stopwords, punctuation, digits, and special characters.

### Feature Engineering
1. **Bag of Words**: Basic word presence features.
2. **Bigrams**: Word pairs included using PMI (Pointwise Mutual Information).
3. **Subjectivity Lexicon**: Word subjectivity based on polarity.
4. **Negation Handling**: Captured negation features.
5. **Comprehensive Features**: Combined bigrams, negations, and subjectivity.

### Models
- **Naïve Bayes**: Trained using different feature functions and cross-validation.
- **Decision Tree**: Also applied, but with lower accuracy.
- **Transformers**: Fine-tuned models (BERT, RoBERTa, DistilBERT) for comparative analysis.

### Cross-Validation
Performed 5-fold and 10-fold cross-validation using comprehensive features (bigrams, negations, subjectivity).

### Evaluation Metrics
- **Accuracy**: Percentage of correctly classified reviews.
- **F-1 Score**: Harmonic mean of precision and recall.

## Results

- **Best Model**: Naïve Bayes with comprehensive features (bigrams, negations, subjectivity) on 40,000 reviews using 10-fold cross-validation achieved an **F-1 score of 85.1%**.
- **Other Highlights**:
  - Model excluding neutral sentiments achieved **accuracy of 81.67%**.
  - Experiments with bigrams, negations, and lexicons showed varying results.

## How to Run

1. **Requirements**:
   - Python 3.x
   - `scikit-learn`
   - `nltk`
   - `transformers`
   - `pandas`
   
2. **Scripts**:
   - **`classifyKaggle.py`**: Contains various feature functions for training models and making predictions.
   - **`classifyKaggle.crossval.py`**: Contains code for cross-validation.
   - **`predictions.csv`**: Contains the predictions from the model.
   
3. **Steps**:
   - Download and place the Kaggle Movie Reviews Dataset in the `/data` folder.
   - Run `classifyKaggle.py` to train the models.
   - Run `classifyKaggle.crossval.py` to perform cross-validation.
   - Use the resulting model to classify sentiments from the test dataset.

4. **Cross-Validation**:
   - Modify the feature function used in `classifyKaggle.crossval.py` to switch between 5-fold and 10-fold cross-validation.
   
## Project Files
- `classifyKaggle.py`: Code for training models with various feature functions.
- `classifyKaggle.crossval.py`: Code for cross-validation.
- `predictions.csv`: CSV file containing the predicted results for the test dataset.

## Conclusion

This project uses a variety of NLP techniques to classify movie reviews into positive and negative sentiments. The Naïve Bayes model with comprehensive features and 10-fold cross-validation achieved the best performance, making it suitable for this classification task.
