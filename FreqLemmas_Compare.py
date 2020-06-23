import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from spacy.tokenizer import Tokenizer


def main():
    print("Processing %s rows in Eng High Freq" % len(
        open("Data/Training/EngHighFreqLemmas.txt", encoding="utf8").readlines()))
    print("Processing %s rows in Spn High Freq" % len(
        open("Data/Training/SpnHighFreqLemmas.txt", encoding="utf8").readlines()))

    EngFreqLemmas = open("Data/Training/EngHighFreqLemmas.txt", encoding="utf8").read()
    SpnFreqLemmas = open("Data/Training/SpnHighFreqLemmas.txt", encoding="utf8").read()

    EngFreqLemmas = EngFreqLemmas.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")
    SpnFreqLemmas = SpnFreqLemmas.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")

    # tokenize english freq list
    nlp_en = spacy.load("en_core_web_sm")
    tokenizer_en = Tokenizer(nlp_en.vocab)
    eng_tokens = tokenizer_en(EngFreqLemmas)
    df_eng = pd.DataFrame()
    df_eng['Token'] = [token.text for token in eng_tokens]
    df_eng['Language'] = 'Eng'
    df_eng.drop_duplicates(keep='first', inplace=True)
    df_eng.to_csv(r'Output/EngHighFreqLemmas.csv', index=None, header=True)

    # tokenize spanish freq list
    nlp_es = spacy.load('es_core_news_sm')
    tokenizer_es = Tokenizer(nlp_es.vocab)
    spn_tokens = tokenizer_es(SpnFreqLemmas)
    df_spn = pd.DataFrame()
    df_spn['Token'] = [token.text for token in spn_tokens]
    df_spn['Language'] = 'Spn'
    df_spn.drop_duplicates(keep='first', inplace=True)
    df_spn.to_csv(r'Output/SpnHighFreqLemmas.csv', index=None, header=True)

    # count unique words in each high freq list
    print("Processing %s unique words in Eng High Freq" % len(df_eng))
    print("Processing %s unique words in Spn High Freq" % len(df_spn))

    # Merge eng and spn to check overlapping
    outer_df = pd.merge(df_eng, df_spn, on='Token', how='outer', indicator='Exist')
    diff_df = outer_df.loc[outer_df['Exist'] == 'both']
    print("Number of tokens in both high frequent lists is %s" % len(diff_df))
    diff_df.to_csv(r'Output/diff_df_High_Freq.csv', index=None, header=True)

    # Exclude the overlapping list from Eng frequent list
    new_df_eng = df_eng[~df_eng['Token'].isin(diff_df['Token'])].dropna()

    # Merge eng and spn together for N-gram training
    df = pd.concat([new_df_eng, df_spn])
    print('Distribution of Language\n', df['Language'].value_counts())

    lang_label = df["Language"]
    lang_X = df["Token"]
    lang_X, lang_label = shuffle(lang_X, lang_label)
    X_train, X_test, y_train, y_test = train_test_split(lang_X, lang_label, test_size=0.25)

    pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1, 5), analyzer='char')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())])

    pipe.fit(X_train, y_train)
    y_predicted = pipe.predict(X_test)
    y_predicted_prob = pipe.predict_proba(X_test)

    # check the performance of the n-gram model
    print(classification_report(y_test, y_predicted))
    print(confusion_matrix(y_test, y_predicted))
    print(accuracy_score(y_test, y_predicted))

    # Assign probability to tokens from opinion articles
    target_OpinionArticles_csv = pd.read_csv('Output/target_full_df_OA.csv', encoding='utf-8')
    target_OpinionArticles_csv_df = pd.DataFrame(target_OpinionArticles_csv)

    target_OpinionArticles_X = target_OpinionArticles_csv_df["Token"]

    y_predicted_OpinionArticles = pipe.predict(target_OpinionArticles_X)
    y_predicted_prob_OpinionArticles = pipe.predict_proba(target_OpinionArticles_X)

    y_predicted_OpinionArticles_df = pd.DataFrame(y_predicted_OpinionArticles, columns=['predicted_Lang'],
                                                  index=target_OpinionArticles_csv_df.index)
    y_predicted_prob_OpinionArticles_df = pd.DataFrame(y_predicted_prob_OpinionArticles,
                                                       columns=['Eng_prob', 'Spn_prob'],
                                                       index=target_OpinionArticles_csv_df.index)
    target_OpinionArticles_csv_df_prob = pd.concat([target_OpinionArticles_csv_df, y_predicted_prob_OpinionArticles_df],
                                                   axis=1)
    target_OpinionArticles_csv_df_prob = pd.concat([target_OpinionArticles_csv_df_prob, y_predicted_OpinionArticles_df],
                                                   axis=1)

    target_OpinionArticles_csv_df_prob.to_csv(r'Output/target_full_df_OA_Prob.csv', index=None, header=True)

    # NACC
    target_NACC_csv = pd.read_csv('Output/target_full_df_NACC.csv', encoding='utf-8')
    target_NACC_csv_df = pd.DataFrame(target_NACC_csv)

    target_NACC_X = target_NACC_csv_df["Token"]

    y_predicted_NACC = pipe.predict(target_NACC_X)
    y_predicted_prob_NACC = pipe.predict_proba(target_NACC_X)

    y_predicted_NACC_df = pd.DataFrame(y_predicted_NACC, columns=['predicted_Lang'],
                                       index=target_NACC_csv_df.index)
    y_predicted_prob_NACC_df = pd.DataFrame(y_predicted_prob_NACC, columns=['Eng_prob', 'Spn_prob'],
                                            index=target_NACC_csv_df.index)
    target_NACC_csv_df_prob = pd.concat([target_NACC_csv_df, y_predicted_prob_NACC_df], axis=1)
    target_NACC_csv_df_prob = pd.concat([target_NACC_csv_df_prob, y_predicted_NACC_df], axis=1)
    target_NACC_csv_df_prob.to_csv(r'Output/target_full_df_NACC_Prob.csv', index=None, header=True)


if __name__ == '__main__':
    main()
