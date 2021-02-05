# CATHETER COLUMNS
ETT_COLUMNS = ["ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]
NGT_COLUMNS = ["NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"]
CVC_COLUMNS = ["CVC - Abnormal", "CVC - Borderline", "CVC - Normal"]
SWAG_COLUMN = ["Swan Ganz Catheter Present"]


# assuming we don't need squeeze and transpose (!!!!!!!!!!!!!!!!!!!!!)
# Catheter Dataframes
ETT_df_submission = pd.DataFrame(ETT_pred, columns = ETT_COLUMNS)
NGT_df_submission = pd.DataFrame(NGT_pred, columns = NGT_COLUMNS)
CVC_df_submission = pd.DataFrame(CVC_pred, columns = CVC_COLUMNS)
SWAG_df_submission = pd.DataFrame(SWAG_pred, columns = SWAG_COLUMN)

# StudyInstanceUID DataFrame
SUID_df_submission = pd.DataFrame(test_files, "StudyInstanceUID")

# concatenated dataframes
catheter_df = [SUID_df_submission, ETT_df_submission, NGT_df_submission, CVC_df_submission, SWAG_df_submission]

# FINAL SUBMISSION DF

# puts IDs in first dataframe index
df_submission = pd.concat(catheter_df)

# SUBMISSION FILE
df_submission.to_csv("submission.csv", index=False)
