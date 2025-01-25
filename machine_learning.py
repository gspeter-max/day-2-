'''
Problem 1: Multiclass Text Classification with Non-Linear Dependencies
You are tasked with building an SVM-based text classification model for an imbalanced, multilingual dataset.

Dataset Details:
Text data: 100,000 samples across 10 languages.
Classes: 15 classes with severe imbalance:
3 classes make up 80% of the data.
The remaining 12 classes account for 20%.
Multilingual tokens:
English (50%), French (20%), Spanish (15%), and 7 other languages (15% combined).
Some samples are code-mixed (e.g., English + French in a single sentence).
Challenge Requirements:

Preprocessing:
Tokenization and feature extraction using TF-IDF or Word2Vec.
Handle code-mixed text effectively.
Model:
Use SVM with kernels (experiment with RBF, polynomial, and custom kernels).
Hyperparameter Tuning:
Optimize C, gamma, and kernel-specific parameters using grid search with stratified cross-validation.
Metrics:
Evaluate with macro F1-score and precision/recall curves for each class.
Stretch Goals:

Develop a custom kernel to handle semantic relationships between languages.
Implement multi-label classification where a sample may belong to multiple classes.
''' 

''' data set '''
import pandas as pd
import random
import numpy as np

# Set seed for reproducibility
random.seed(42)

# Constants
num_samples = 100  # Total samples
languages = ['English', 'French', 'Spanish', 'Other_Language_1', 'Other_Language_2', 'Other_Language_3', 
             'Other_Language_4', 'Other_Language_5', 'Other_Language_6', 'Other_Language_7']
languages_weights = [0.5, 0.2, 0.15, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025, 0.025]

classes = [f'Class_{i}' for i in range(1, 16)]
class_weights = [0.8, 0.1, 0.1] + [0.05]*12  # 3 classes make up 80%, 12 classes are minor
language_mix_probabilities = {
    'English': 0.5, 'French': 0.2, 'Spanish': 0.15, 'Other_Language_1': 0.025,
    'Other_Language_2': 0.025, 'Other_Language_3': 0.025, 'Other_Language_4': 0.025,
    'Other_Language_5': 0.025, 'Other_Language_6': 0.025, 'Other_Language_7': 0.025
}

# Generate samples
data = []

for _ in range(num_samples):
    # Randomly select class based on class weights
    class_choice = random.choices(classes[:3], weights=[80, 10, 10])[0] if random.random() < 0.8 else random.choices(classes[3:], weights=[5]*12)[0]
    
    # Randomly choose languages and simulate code-mixing (for 10% of cases)
    language_choice = random.choices(languages, weights=languages_weights)[0]
    if random.random() < 0.1:  # 10% chance of code-mixing
        code_mixed_language = random.choice([l for l in languages if l != language_choice])
        text_sample = f"Sentence in {language_choice} + {code_mixed_language}"  # Code-mixed sentence
    else:
        text_sample = f"Sentence in {language_choice}"  # Non-code-mixed sentence
    
    # Append the sample to data list
    data.append([text_sample, class_choice])

# Create DataFrame
df = pd.DataFrame(data, columns=['text', 'class'])

''' 1st solution ''' 
import pandas as pd 
import nltk 
import spacy 
from nltk.tokenize import word_tokenize

eng_model = spacy.load('en_core_web_sm')
fra_model = spacy.load('fr_core_news_sm')

def text_tokens(text , language_choice = 'eng'):
  lists = [] 
  if language_choice == 'eng':
    docs = eng_model(text)
  elif language_choice == 'fra':
    docs = fra_model(text)
  else:
    docs = eng_model(text)
  return [token.text.lower() for token in docs  if not token.is_stop and not token.is_punct]

df['more_accuracte_text'] = df['text'].apply(text_tokens)

def do_tokenization(text):
  words = nltk.word_tokenize(text)
  return words 
df['simple_reuslt']  = df['text'].apply(lambda x: do_tokenization(x))

from sklearn.feature_extraction.text import TfidfVectorizer
tifid_model = TfidfVectorizer(
    tokenizer=lambda x : x.split(),stop_words='english'
)
x_tifid  = tifid_model.fit_transform(df['text'])
df_temp = pd.DataFrame(
    x_tifid.toarray() , columns = tifid_model.get_feature_names_out()
)




