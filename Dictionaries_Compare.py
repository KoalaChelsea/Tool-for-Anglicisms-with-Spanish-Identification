import spacy
import pandas as pd
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():

    print("Processing %s rows in lemario-espanol" % len(
        open("Data/Dictionaries/lemario-espanol-2002-10-25.txt", encoding="utf8").readlines()))
    print("Processing %s rows in lemario-utf8" % len(
        open("Data/Dictionaries/lemario-utf8.txt", encoding="utf8").readlines()))

    SpnDist1 = open("Data/Dictionaries/lemario-espanol-2002-10-25.txt", encoding="utf8").read()
    SpnDist2 = open("Data/Dictionaries/lemario-utf8.txt", encoding="utf8").read()

    SpnDist1 = SpnDist1.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")
    SpnDist2 = SpnDist2.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")

    # tokenize spanish freq list
    nlp_es = spacy.load('es_core_news_sm')
    tokenizer_es = Tokenizer(nlp_es.vocab)
    spn_tokens1 = tokenizer_es(SpnDist1)
    spn_tokens2 = tokenizer_es(SpnDist2)
    df_spn1 = pd.DataFrame()
    df_spn2 = pd.DataFrame()
    df_spn1['Token'] = [token.text for token in spn_tokens1]
    df_spn2['Token'] = [token.text for token in spn_tokens2]
    df_spn1.drop_duplicates(keep='first', inplace=True)
    df_spn2.drop_duplicates(keep='first', inplace=True)
    df_spn1.to_csv(r'Output/lemario-espanol.csv', index=None, header=True)
    df_spn2.to_csv(r'Output/lemario-utf8.csv', index=None, header=True)

    # count unique words in each dictionary
    print("Processing %s unique words in lemario-espanol-2002-10-25" % len(df_spn1))
    print("Processing %s unique words in lemario-utf8" % len(df_spn2))


if __name__ == '__main__':
    main()
