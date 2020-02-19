import pandas as pd

def main():
    # read the file after tokenizing
    SpacyTokenizer_csv_OA = pd.read_csv('Output/OpinionArticles-TASI.csv', encoding='utf-8')
    SpacyTokenizer_csv_NACC = pd.read_csv('Output/NACCf27k-text-TASI.csv', encoding='utf-8')
    SpacyTokenizer_csv_df_OA = pd.DataFrame(SpacyTokenizer_csv_OA)
    SpacyTokenizer_csv_df_NACC = pd.DataFrame(SpacyTokenizer_csv_NACC)
    print("Processing %s word in spacy tokenizer document (OpinionArticles)" % len(SpacyTokenizer_csv_df_OA))
    print("Processing %s word in spacy tokenizer document (NACC)" % len(SpacyTokenizer_csv_df_NACC))


    # read Goldstandard file
    GoldStandard_csv_OA = pd.read_csv('Data/OpinionArticlesRetokenized-GS.csv', encoding='utf-8')
    GoldStandard_tsv_NACC = pd.read_csv('Data/NACC27k-GoldStandard.tsv', sep='\t', encoding='utf-8')
    GoldStandard_csv_df_OA = pd.DataFrame(GoldStandard_csv_OA)
    GoldStandard_tsv_df_NACC = pd.DataFrame(GoldStandard_tsv_NACC)
    print("Processing %s word in gold standard document (OpinionArticles)" % len(GoldStandard_csv_df_OA))
    print("Processing %s word in gold standard document (NACC)" % len(GoldStandard_tsv_df_NACC))


    # check the difference between two files of OpinionArticles
    outer_df_OA = pd.merge(SpacyTokenizer_csv_df_OA, GoldStandard_csv_df_OA, on='Token', how='outer', indicator='Exist')
    diff_df_OA = outer_df_OA.loc[outer_df_OA['Exist'] != 'both']
    print("Number of non-matching token for OpinionArticles is %s" % len(diff_df_OA))
    diff_df_OA.to_csv(r'Output/diff_df_OA.csv', index=None, header=True)

    # check the difference between two files of NACC
    outer_df_NACC = pd.merge(SpacyTokenizer_csv_df_NACC, GoldStandard_tsv_df_NACC, on='Token', how='outer', indicator='Exist')
    diff_df_NACC = outer_df_NACC.loc[outer_df_NACC['Exist'] != 'both']
    print("Number of non-matching token for NACC is %s" % len(diff_df_NACC))
    diff_df_NACC.to_csv(r'Output/diff_df_NACC.csv', index=None, header=True)


    # df of interested tokens of OpinionArticles
    target_token_OA = SpacyTokenizer_csv_df_OA[SpacyTokenizer_csv_df_OA.Anglicism.isnull()]
    target_full_df_OA = pd.merge(target_token_OA, GoldStandard_csv_df_OA, on='Token', how='left')
    target_full_df_OA.drop(['New_Index', 'Anglicism_x', 'Language_x'], axis=1, inplace=True)
    target_full_df_OA.rename(columns={'Anglicism_y': 'Anglicism'}, inplace=True)
    target_full_df_OA.rename(columns={'Language_y': 'Language'}, inplace=True)
    target_full_df_OA.drop_duplicates(keep='first', inplace=True)
    print("Processing %s word in target document (OpinionArticles)" % len(target_full_df_OA))
    print(target_full_df_OA.columns)

    # df of interested tokens of NACC
    target_token_NACC = SpacyTokenizer_csv_df_NACC[SpacyTokenizer_csv_df_NACC.Anglicism.isnull()]
    target_full_df_NACC = pd.merge(target_token_NACC, GoldStandard_tsv_df_NACC, on='Token', how='left')
    target_full_df_NACC.drop(['Anglicism_x', 'Language_x'], axis=1, inplace=True)
    target_full_df_NACC.rename(columns={'Anglicism_y': 'Anglicism'}, inplace=True)
    target_full_df_NACC.rename(columns={'Language_y': 'Language'}, inplace=True)
    target_full_df_NACC.drop_duplicates(keep='first', inplace=True)
    print("Processing %s word in target document (NACC)" % len(target_full_df_NACC))
    print(target_full_df_NACC.columns)

    # write to csv
    target_full_df_OA.to_csv(r'Output/target_full_df_OA.csv', index=None, header=True)
    target_full_df_NACC.to_csv(r'Output/target_full_df_NACC.csv', index=None, header=True)


if __name__ == '__main__':
    main()