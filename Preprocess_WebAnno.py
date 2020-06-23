import pandas as pd
import csv
def main():

    # Read in gold standard of OpinionArticles from Webanno
    with open("Data/WebAnno/OpinionArticles-text.txt") as input:
        with open("Data/WebAnno/OpinionArticles-text-cleaned.txt", "w") as output:
            for line in input:
                if not (line.startswith("#Text=") or line.startswith("#T_SP=") or line.startswith("#FORMAT=")):
                    if not line.strip(): continue  # skip the empty line
                    output.write(line)

    OpinionArticles_Webanno = pd.read_csv("Data/WebAnno/OpinionArticles-text-cleaned.txt", delimiter="\t", header=None,
                                          quoting=csv.QUOTE_NONE, encoding='utf-8', index_col=False,
                                          names=["text-token", "token-char", "Token", "star", "Adapted", "EngCategory"])
    OpinionArticles_Webanno.drop(["star"], axis=1, inplace=True)

    print("Processing %s tokens from WebAnno (OpinionArticles)" % len(OpinionArticles_Webanno))
    print(OpinionArticles_Webanno.head())


    # check whether the all the tokens with label are included in SpaCy tokens
    SpacyTokenizer_csv_OA = pd.read_csv('Output/OpinionArticles-TASI.csv', encoding='utf-8')
    SpacyTokenizer_csv_df_OA = pd.DataFrame(SpacyTokenizer_csv_OA)
    print("Processing %s words in spacy tokenizer document (OpinionArticles)" % len(SpacyTokenizer_csv_df_OA))


    # check the  OpinionArticles
    outer_df_OA = pd.merge(OpinionArticles_Webanno, SpacyTokenizer_csv_df_OA, on='Token', how='outer', indicator='Exist')
    diff_df_OA = outer_df_OA.loc[outer_df_OA['Exist'] != 'both']
    print("Number of non-matching token for OpinionArticles is %s" % len(diff_df_OA))
    diff_df_OA.to_csv(r'Data/WebAnno/diff-token.csv', index=None, header=True)
    print(diff_df_OA)





if __name__ == '__main__':
    main()