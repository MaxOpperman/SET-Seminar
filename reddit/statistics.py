import os

import numpy as np
import pandas as pd


# Import the data from the computed, pre-labelled, and diversity csv files.
# Also find the number of files in the labelling output folder
def compute_statistics():
    computed_df = pd.read_csv("./out/redditManualExtractedComputed.csv")
    pre_labelled_df = pd.read_csv("./labels/redditManualExtractedLabelled2.csv")
    diversity = pd.read_csv("./out/redditManualExtractedDiversity.csv", header=[0, 1], index_col=[0, 1])
    n = sum(1 for f in os.listdir("./labels/out/") if os.path.isfile(os.path.join("./labels/out/", f)))
    last_labelled_df = pd.read_csv(f"./labels/out/redditManualExtractedLabelled{n}.csv")

    # Compute and print the statistics of the comments on the computed csv file
    total_no_ageism = computed_df['NoAgeism'].sum()
    total_confirm_ageism = computed_df['ConfirmAgeism'].sum()
    total_favor_ageism = computed_df['FavorAgeism'].sum()
    total_unusable = computed_df['Unusable'].sum()
    total_comments = computed_df['Comments'].sum()

    # print data about the last labelled comments
    print(f"\nNo ageism: {total_no_ageism} ({total_no_ageism / total_comments})\n"
          f"Confirm ageism: {total_confirm_ageism} ({total_confirm_ageism / total_comments})\n"
          f"Favor ageism: {total_favor_ageism} ({total_favor_ageism / total_comments})\n"
          f"Unusable: {total_unusable} ({total_unusable / total_comments})\n"
          f"total: {total_no_ageism + total_confirm_ageism + total_favor_ageism + total_unusable} -"
          f" {total_comments}")

    # Compute and print the statistics of the comments on the pre-labelled csv file
    print("\nFor pre-labelled DF:")
    total_no_ageism = len(pre_labelled_df[pre_labelled_df["Label"] == "Doesn't think ageism exists."])
    total_confirm_ageism = len(pre_labelled_df[pre_labelled_df["Label"] == "Acknowledges ageism exists and has a "
                                                                           "negative association with it."])
    total_favor_ageism = len(pre_labelled_df[pre_labelled_df["Label"] == "Favors ageism."])
    total_unusable = len(pre_labelled_df[pre_labelled_df["Label"] == "Unusable."])
    total_comments = len(pre_labelled_df.index)
    print(f"\nNo ageism: {total_no_ageism} ({total_no_ageism / total_comments})\n"
          f"Confirm ageism: {total_confirm_ageism} ({total_confirm_ageism / total_comments})\n"
          f"Favor ageism: {total_favor_ageism} ({total_favor_ageism / total_comments})\n"
          f"Unusable: {total_unusable} ({total_unusable / total_comments})\n"
          f"total: {total_no_ageism + total_confirm_ageism + total_favor_ageism + total_unusable} -"
          f" {total_comments}")

    # print stuff about the diversity aspects
    print(diversity)
    diversity = diversity[diversity.index.get_level_values(1) != 'Unusable.']
    for index, row in diversity.iterrows():
        print(row)

    # print stuff about the sentiment analyses
    sentiment_df = last_labelled_df.groupby(["Label"]).agg({
        "ID": "count",
        "Sentiment1": ["sum", "mean", "std", "median"],
        "Sentiment2": ["sum", "mean", "std", "median"],
    })
    print(sentiment_df)
    # write the sentiment analyses to a csv file
    with open(f"./out/redditManualExtractedSentiment.csv", "w+", newline="\n", encoding="utf-8") as sent_file:
        sentiment_df.to_csv(sent_file)

