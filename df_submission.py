
# Create Submission df
df_submission = pd.DataFrame(np.squeeze(pred)).transpose()
df_submission.rename(columns=dict(zip([str(i) for i in range(12)], label_cols)))
df_submission["StudyInstanceUID"] = test_files

# SUBMISSION FILE
df_submission.to_csv("submission.csv", index=False)


#ETT
pd.DataFrame(pred, columns = ETT_COLUMNS)

Where ETT_COLUMNS = df_train["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]

^ That would be a filtered dataframe of just those columns and their content though. Not just the 4 column labels.


# Concatenation
pd.concat([pred_df], axis = 1)
