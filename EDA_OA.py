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
    return df


def main():
    # read in opinionArticles
    text_OpinionArticles = open("Data/OpinionArticles-text.txt", encoding="utf8").read()

    # clean text
    clean_text_OpinionArticles = cleanText(text_OpinionArticles)

    # Default read text in both english and spanish
    nlp_es = spacy.load('es_core_news_sm')
    nlp_en = spacy.load("en_core_web_sm")

    # Tokenize text using custom tokenizer
    nlp_es.tokenizer = custom_tokenizer_modified(nlp_es)
    nlp_en.tokenizer = custom_tokenizer_modified(nlp_en)

    # spacy text
    doc_OpinionArticles_es = nlp_es(clean_text_OpinionArticles)
    doc_OpinionArticles_en = nlp_en(clean_text_OpinionArticles)

    # write token into data frame
    text_OpinionArticles_df_es = custom_tokenizer_to_df(nlp_es, doc_OpinionArticles_es)
    text_OpinionArticles_df_en = custom_tokenizer_to_df(nlp_en, doc_OpinionArticles_en)

    # update user on length of tokens
    print("Processing %s-word OpinionArticles using spanish tokenizer" % len(doc_OpinionArticles_es))
    print("Processing %s-word OpinionArticles using english tokenizer" % len(doc_OpinionArticles_en))

    # Check difference using different tokenizer
    text_OpinionArticles_df_es.drop_duplicates(keep='first', inplace=True)
    text_OpinionArticles_df_en.drop_duplicates(keep='first', inplace=True)
    outer_df_OA = pd.merge(text_OpinionArticles_df_es, text_OpinionArticles_df_en,
                           on=['Token', 'Lemma'], how='outer', indicator='Exist')
    diff_df_OA = outer_df_OA.loc[outer_df_OA['Exist'] != 'both'].sort_values(by=['Token'])
    print("Number of non-matching token for OpinionArticles using different tokenizer is %s" % len(diff_df_OA))
    diff_df_OA.to_csv(r'Output/diff_token_es_en.csv', index=None, header=True)


if __name__ == '__main__':
    main()
