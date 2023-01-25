import csv
import os
import re

import pandas as pd
import praw
from matplotlib import pyplot as plt
from dotenv import load_dotenv

from reddit.categorize_sklearn import classify_sentiment
from reddit.functions import get_posts, get_comments, read_manual_posts

# Create an app: https://www.reddit.com/prefs/apps
# Use http://localhost:8080 as redirect uri
from reddit.categorize_gensim import classify_comments
from reddit.statistics import compute_statistics

load_dotenv()
username = os.getenv("REDDIT_USERNAME")
password = os.getenv("PASSWORD")
clientid = os.getenv("CLIENTID")
clientsecret = os.getenv("CLIENTSECRET")


def input_number(message: str):
    try:
        user_input = int(input(message))
    except ValueError:
        return None
    else:
        return user_input


def __init__():
    outfile_name = input("Enter a CSV filename to output the data to (e.g., reddit-data.csv)\r\n")
    pickle_folder = input("Enter a folder where the pickled files are stored (e.g. 'auto' for './pickle/auto/')"
                          "\r\n").strip("/")
    if pickle_folder == '':
        pickle_folder = 'auto'

    num_topics = input_number("Enter the number of topics to categorize in (leave empty to get the optimal value, "
                              "this might take a while)\r\n")
    train_set_size = None
    if num_topics is None:
        train_set_size = input_number("Enter the number of comments to base the number of topics on (default 1000)\r\n")
        if train_set_size is None:
            train_set_size = 1000
    if outfile_name.endswith(".csv"):
        outfile_name = outfile_name[:outfile_name.rindex(".csv")]

    file_exists = os.path.exists(f"{outfile_name}.csv")

    reddit = praw.Reddit(client_id=clientid,
                         client_secret=clientsecret,
                         password=password,
                         user_agent='Reddit search data bot extractor by /u/' + username + '',
                         username=username)
    print("Authentication for " + str(reddit.user.me()) + " is verified. Proceeding.\r\n")

    if "statistic" in outfile_name.lower():
        compute_statistics()
        return 0

    elif not file_exists:
        search = input("Enter a search (e.g., 'how do you') or multiple searches delimited with commas:\r\n")
        sort_sub = input("How do you want to sort results? Enter relevance, hot, top, new, or comments.\r\n")
        filter_sub = input("Do you want to restrict to a certain subreddit? Enter 'Yes' or 'No'.\r\n")

        # get the posts with the parameters above
        get_posts(outfile_name, reddit, search, sort_sub, filter_sub)

    elif "comments" in outfile_name.lower():
        df = pd.read_csv(f'{outfile_name}.csv')
        if not os.path.exists(f'./pickle/{pickle_folder}'):
            os.makedirs(f'./pickle/{pickle_folder}')
        # classify the sentiment
        # classify_sentiment(df["Body"].tolist(), df["NLTK sentiment"].tolist(), pickle_folder, train_set_size)
        # write the classifications
        (df['Dominant_Topic'], df['Perc_Contribution'], df['Topic_Keywords']) = \
            classify_comments(df["Body"].tolist(), pickle_folder, num_topics, train_set_size)
        with open(f"{outfile_name}Categories.csv", "w+", newline="\n", encoding="utf-8") as categories_file:
            df.to_csv(categories_file)
            print(f"Wrote categories to ./{outfile_name}Categories.csv")
        return 0

    elif "manual" in outfile_name.lower():
        if not os.path.exists('./out'):
            os.makedirs('./out')
        outfile_name = read_manual_posts(outfile_name, reddit)

    get_comments(outfile_name, reddit, train_set_size=train_set_size, num_topics=num_topics,
                 pickle_folder=pickle_folder)
