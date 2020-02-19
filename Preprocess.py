# Using Python 3.6.5
# TO DO
## Note multi-word anglicisms as a unit with I,O,B tagging

import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
from spacy.matcher import Matcher
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import re


# clean text before spacy
def cleanText(text):
    # get rid of new line terminator
    text = text.strip().replace("\n", " ").replace("\r", " ").replace("\r\n", " ").replace("  ", " ")
    return text


# deal with more symbols to seperate tokens
def custom_tokenizer_modified(nlp):
    # spacy defaults: when the standard behaviour is required, they
    # need to be included when subclassing the tokenizer
    infix_re = re.compile(r'''[.\,\?\!\:\...\‘\’\`\“\”\"\'\/~]''')
    extended_prefixes = tuple(list(nlp.Defaults.prefixes) + ["-"])
    prefix_re = compile_prefix_regex(extended_prefixes)
    extended_suffixes = tuple(list(nlp.Defaults.suffixes) + ["-"])
    suffix_re = compile_suffix_regex(extended_suffixes)

    # extending the default url regex
    url = URL_PATTERN
    url_re = re.compile(url)
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=url_re.match
                     )


# write all the tokens extracted from text into a data frame
def custom_tokenizer_to_df(nlp, doc):
    # Initialize the Matcher with a vocab
    matcher = Matcher(nlp.vocab)

    ###############################################################
    # Add pattern for valid hashtag, i.e. '#' plus any ASCII token
    matcher.add("HASHTAG", None, [{"ORTH": "#"}, {"IS_ALPHA": True}])

    # Register token extension for hashtag
    Token.set_extension("is_hashtag", default=False, force=True)

    # Fit in text in matcher
    matches = matcher(doc)

    # Find hashtag and merge, assign hashtag label
    hashtags = []
    for match_id, start, end in matches:
        if doc.vocab.strings[match_id] == "HASHTAG":
            hashtags.append(doc[start:end])
    with doc.retokenize() as retokenizer:
        for span in hashtags:
            retokenizer.merge(span)
            for token in span:
                token._.is_hashtag = True
    ##############################################################

    ##############################################################
    # Find number and merge, assign number label
    # Add pattern for valid hashtag, i.e. '#' plus any ASCII token
    matcher.add("LONG_NUMBER", None, [{"IS_DIGIT": True}, {"ORTH": ','}, {"IS_DIGIT": True}])
    matcher.add("LONG_NUMBER", None, [{"IS_DIGIT": True}, {"ORTH": '.'}, {"IS_DIGIT": True}])

    # Register token extension for hashtag
    Token.set_extension("is_long_number", default=False, force=True)

    # Fit in text in matcher
    matches = matcher(doc)

    long_number = []
    for match_id, start, end in matches:
        if doc.vocab.strings[match_id] == "LONG_NUMBER":
            long_number.append(doc[start:end])
    with doc.retokenize() as retokenizer:
        for span in long_number:
            retokenizer.merge(span)
            for token in span:
                token._.is_long_number = True
    ##############################################################

    for i, token in enumerate(doc):
        if token._.is_hashtag:
            token.tag_ = 'Hashtag'
        if token.like_url:
            token.tag_ = 'URL'
        if token.like_email:
            token.tag_ = 'Email'
        if token.is_stop:
            token.tag_ = 'Stop Word'
        if token.like_num:
            token.tag_ = 'Number'
        if token._.is_long_number:
            token.tag_ = 'Number'
        if token.is_punct:
            token.tag_ = 'Punctuation'

    # Write the tokens to data frame
    df = pd.DataFrame()
    df['Token'] = [token.text for token in doc]
    df['POS'] = [token.pos_ for token in doc]
    df['NE'] = [token.ent_iob_ for token in doc]
    df['Lemma'] = [token.lemma_ for token in doc]
    df['Tag'] = [token.tag_ for token in doc]
    df['Language'] = np.nan
    df['Candidate'] = True
    df['Anglicism'] = np.nan
    return df


def filter_noninterested_text(df):
    # filter out non interested text by three criteria below
    for i in df.Token.index:
        # Ignore bad words
        if df.loc[i, "POS"] in ['Hashtag', 'URL', 'Email', 'Stop Word', 'Number', 'Punctuation']:
            df.loc[i, 'Language'] = df.loc[i, 'Tag']
            df.loc[i, 'Candidate'] = False
            df.loc[i, 'Anglicism'] = False
        # Ignore any POS tags that are not Noun, VERB, or Adjective
        elif df.loc[i, "POS"] not in ['NOUN', 'VERB', 'ADJ']:
            df.loc[i, 'Language'] = df.loc[i, 'POS']
            df.loc[i, 'Candidate'] = False
            df.loc[i, 'Anglicism'] = False
        # Ignore NEs
        elif df.loc[i, 'NE'] != 'O':
            df.loc[i, 'Language'] = 'Name Entity'
            df.loc[i, 'Candidate'] = False
            df.loc[i, 'Anglicism'] = False
    return df


def main():
    # Default read text in spanish
    nlp = spacy.load('es_core_news_sm')

    # Tokenize text using custom tokenizer
    nlp.tokenizer = custom_tokenizer_modified(nlp)

    # Samples to run in python console or testing

    text_OpinionArticles = open("Data/OpinionArticles-text.txt", encoding="utf8").read()
    text_NACC = open("Data/NACC27k-text.txt", encoding="utf8").read()

    # clean text
    clean_text_OpinionArticles = cleanText(text_OpinionArticles)
    clean_text_NACC = cleanText(text_NACC)

    # spacy text
    doc_OpinionArticles = nlp(clean_text_OpinionArticles)
    doc_NACC = nlp(clean_text_NACC)

    # write token into data frame
    text_OpinionArticles_df = custom_tokenizer_to_df(nlp, doc_OpinionArticles)
    text_NACC_df = custom_tokenizer_to_df(nlp, doc_NACC)

    # update user on length of tokens
    print("Processing %s word document (OpinionArticles)" % len(doc_OpinionArticles))
    print("Processing %s word document (NACC)" % len(doc_NACC))

    # Filter out non interested tokens by assigning label
    filter_noninterested_text(text_OpinionArticles_df)
    filter_noninterested_text(text_NACC_df)

    # write df to csv
    text_OpinionArticles_df.to_csv(r'Output/OpinionArticles-TASI.csv', index=None, header=True)
    text_NACC_df.to_csv(r'Output/NACCf27k-text-TASI.csv', index=None, header=True)

    # open annotated file
    # subprocess.call(['open',r'Data/OpinionArticles-TASI.csv'])


if __name__ == '__main__':
    main()
