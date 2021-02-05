
pd.DataFrame(pred, columns = ETT_COLUMNS)

Where ETT_COLUMNS = df_train["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]

^ That would be a filtered dataframe of just those columns and their content though. Not just the 4 column labels.
