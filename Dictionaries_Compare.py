import spacy
import pandas as pd
from spacy.tokenizer import Tokenizer


def main():

    # Open three dictionaries: lemario-espanol-2002-10-25.txt, lemario-utf8.txt, Spn High Freq
    # count number of lines(words) in dictionaries
    print("Processing %s rows in lemario-espanol" % len(
        open("Data/Dictionaries/lemario-espanol-2002-10-25.txt", encoding="utf8").readlines()))
    print("Processing %s rows in lemario-utf8 including multiple-word and dash-word" % len(
        open("Data/Dictionaries/lemario-utf8.txt", encoding="utf8").readlines()))
    print("Processing %s rows in Spn High Freq" % len(
        open("Data/Training/SpnHighFreqLemmas.txt", encoding="utf8").readlines()))

    SpnFreqLemmas = open("Data/Training/SpnHighFreqLemmas.txt", encoding="utf8").read()
    SpnDist1 = open("Data/Dictionaries/lemario-espanol-2002-10-25.txt", encoding="utf8").read()

    # Spn High Freq and lemario-espanol-2002-10-25.txt only has one word each line
    SpnFreqLemmas = SpnFreqLemmas.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")
    SpnDist1 = SpnDist1.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")

    # lemario-utf8.txt includes 2-word words, - before or after words
    SpnDistOriginal = open("Data/Dictionaries/lemario-utf8.txt", encoding="utf8").readlines()

    # write two files depending on whether more than one sub-word in a word
    with open('Data/Dictionaries/lemario_multiple_words.txt', 'w', encoding='utf8') as multiple, \
            open('Data/Dictionaries/lemario_one_word.txt', 'w', encoding='utf8') as one, \
            open('Data/Dictionaries/lemario_dash_word.txt', 'w', encoding='utf8') as dash:
        for index, value in enumerate(SpnDistOriginal):
            words = value.split()
            number_of_words = len(words)
            if number_of_words != 1:
                multiple.write(value)
            else:
                if '-' in str(words):
                    dash.write(value)
                else:
                    one.write(value)

    print("Processing %s rows in one word from lemario-utf8.txt" % len(
        open("Data/Dictionaries/lemario_one_word.txt", encoding="utf8").readlines()))
    SpnDist2 = open("Data/Dictionaries/lemario_one_word.txt", encoding="utf8").read()
    SpnDist2 = SpnDist2.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")

    # tokenize spanish freq list
    nlp_es = spacy.load('es_core_news_sm')
    tokenizer_es = Tokenizer(nlp_es.vocab)
    spn_tokens_freq = tokenizer_es(SpnFreqLemmas)
    spn_tokens1 = tokenizer_es(SpnDist1)
    spn_tokens2 = tokenizer_es(SpnDist2)

    df_spn_freq = pd.DataFrame()
    df_spn1 = pd.DataFrame()
    df_spn2 = pd.DataFrame()

    df_spn_freq['Token'] = [token.text for token in spn_tokens_freq]
    df_spn1['Token'] = [token.text for token in spn_tokens1]
    df_spn2['Token'] = [token.text for token in spn_tokens2]
    df_spn_freq.drop_duplicates(keep='first', inplace=True)
    df_spn1.drop_duplicates(keep='first', inplace=True)
    df_spn2.drop_duplicates(keep='first', inplace=True)

    df_spn1.to_csv(r'Output/lemario-espanol.csv', index=None, header=True)
    df_spn2.to_csv(r'Output/lemario-utf8_one_word.csv', index=None, header=True)

    # count unique words in each dictionary
    print("Processing %s unique words in Spn High Freq" % len(df_spn_freq))
    print("Processing %s unique words in lemario-espanol-2002-10-25" % len(df_spn1))
    print("Processing %s unique words in one word from lemario-utf8.txt" % len(df_spn2))

    # Merge eng and spn to check overlapping
    outer_df = pd.merge(df_spn1, df_spn2, on='Token', how='outer', indicator='Exist')
    same_df = outer_df.loc[outer_df['Exist'] == 'both']
    diff_df = outer_df.loc[outer_df['Exist'] != 'both']
    print("Number of same tokens in two dictionaries is %s" % len(same_df))
    print("Number of different tokens in two dictionaries is %s" % len(diff_df))
    diff_df.to_csv(r'Output/same_df_dictionaries.csv', index=None, header=True)
    diff_df.to_csv(r'Output/diff_df_dictionaries.csv', index=None, header=True)


if __name__ == '__main__':
    main()
