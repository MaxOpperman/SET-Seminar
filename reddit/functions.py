import difflib
import os
import re
from googletrans import Translator

import nltk
import csv
import datetime

import numpy as np
import pandas as pd
from nltk import tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from afinn import Afinn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from tqdm import tqdm

from helpers import countries
from reddit.categorize_gensim import classify_comments
from reddit.categorize_sklearn import classify_sentiment


# This function writes the headers for the file.
def write_headers(f):
    f.writerow(["Number", "Keyword", "Title", "Score", "Comments", "URL", "Domain", "Permalink", "ID", "Subreddit",
                "CreatedDate"])


# This function writes data for a submission
def write_fields(f, start_num, submission, search):
    f.writerow([start_num, search.strip(), submission.title,
                submission.score, submission.num_comments,
                submission.url, submission.domain, submission.permalink, submission.id,
                submission.subreddit, datetime.datetime.utcfromtimestamp(submission.created).strftime('%m-%d-%Y')])


# This function writes data for a comment
def write_comment_fields(f, row, comment, comment_body, sid, afn, classification):
    f.writerow([row["Number"], row["Subreddit"], row["Title"], comment.id, comment_body, comment.author,
                comment.author_flair_text, comment.is_submitter, comment.link_id, comment.parent_id,
                comment.edited, comment.score,
                datetime.datetime.utcfromtimestamp(comment.created_utc).strftime('%m-%d-%Y'),
                sid.polarity_scores(comment_body)['compound'], afn.score(comment_body), classification])


# Preprocess the texts by lowercasing, tokenizing, and removing stopwords
def preprocess(text):
    # Lowercase the text
    text = re.sub(r'^>.*\n?|http\S+|[^a-zA-Z0-9 \n.]', '', text.lower(), flags=re.MULTILINE)

    # Tokenize the text
    tokens = tokenize.word_tokenize(text)
    # when looking if somebody is in favor or against an argument the negating words are important
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    stop_words.remove('nor')
    stop_words.remove('no')
    stop_words.remove('t')

    # also stem the data
    lem = WordNetLemmatizer()
    filtered_tokens = [lem.lemmatize(w) for w in tokens if not w.lower() in stop_words]

    return filtered_tokens


# Program to find most frequent element in a list
def most_frequent(element_list):
    dictionary = {}
    count, itm = 0, ''
    for item in reversed(element_list):
        dictionary[item] = dictionary.get(item, 0) + 1
        if dictionary[item] >= count:
            count, itm = dictionary[item], item
    return itm


def get_posts(outfile_name, reddit, search, sort_sub, filter_sub):
    # split the search string into a list of keywords
    search_list = search.split(',')

    if filter_sub.lower() == "yes":
        # prompt the user to enter a list of subreddit names
        subreddit = input("Enter the subreddit names delimited with commas (i.e., BigSEO):\r\n")
        subreddit_list = subreddit.split(',')
        # create a new csv file for the results
        file = open(f"{outfile_name}.csv", "w+", newline="\n", encoding="utf-8")
        f = csv.writer(file)
        # write the headers to the csv file
        write_headers(f)
        # loop through each subreddit in the subreddit list
        for subs in subreddit_list:
            for search_item in search_list:
                start_num = 0
                # search for the keyword in the current subreddit
                for submission in reddit.subreddit(subs.strip()).search(search_item, sort=sort_sub):
                    start_num += 1
                    # write the relevant information to the csv file
                    write_fields(f, start_num, submission, search)
                print(f"Writing out posts results for the search '{search_item.strip()}' in 'r/{subs.strip()}'\r\n")
            # close the csv file
            file.close()
    else:
        # create a new csv file and id file for the results
        file = open(f"{outfile_name}.csv", "w+", newline="\n", encoding="utf-8")
        ids_file = open(f"./newIds.csv", "w+", newline="\n", encoding="utf-8")
        f = csv.writer(file)
        f_ids = csv.writer(ids_file)
        # write the headers to the csv file
        write_headers(f)
        # write headers to the ID file
        f_ids.writerow(["id", "comment", "title"])
        seen = set()
        for search_item in search_list:
            start_num = 0
            # search for the keyword in the "all" subreddit
            for submission in reddit.subreddit('all').search(search_item.lower(), sort=sort_sub):
                if submission.id in seen:
                    print(f"Row with ID '{submission.id}' already seen.")
                    continue
                seen.add(submission.id)
                start_num += 1
                # write the relevant information to the csv file
                write_fields(f, start_num, submission, search)
                # write the submission ID, the title of the submission and the comment of the submission to the ID file
                comment = get_region(str(submission.subreddit), countries.get_all(), None)
                if comment is not None:
                    print(comment, submission.subreddit)
                f_ids.writerow([submission.id, '' if comment is None else f'Region: {comment}', submission.title])
            print("Writing out posts results for the search '" + search_item.strip() + "' in 'r/all'\r\n")
        # close the csv file
        file.close()
        ids_file.close()


def get_region(text, all_regions, post_region):
    # tokenize the text into sentences
    sentences = tokenize.sent_tokenize(text)
    # ensure that there are no questions asked about the languages
    no_questions = "".join([f" {s.lower()} " for s in sentences if not s.endswith('?')])
    # get a list of regions that are in the text
    regions = list(set([reg for reg in all_regions if reg in no_questions]))
    if len(regions) > 1:
        # check if the regions are 'eu' and 'europe'
        if len(regions) == 2 and ' eu' in regions and 'europe' in regions:
            return 'eu'
        # check if the regions are 'german' and 'germany'
        elif len(regions) == 2 and 'german' in regions and 'germany' in regions:
            return 'germany'
        # ask the user to select the correct region from the list of regions found in the text
        return str(difflib.get_close_matches(input(f"What region '{regions}' is the following comment about?\n "
                                                   f"{no_questions}"), all_regions)[0]).replace(" ", "")
    elif len(regions) == 1:
        # check if the region is 'german' and return 'germany'
        if 'german' in regions:
            return 'germany'
        # check if the region is ' eu' and return 'eu'
        if ' eu' in regions:
            return 'eu'
        # return the region found
        return regions[0]
    else:
        # return the default region if no regions found
        return post_region


def read_manual_posts(outfile_name, reddit):
    # open the manual input file for reading
    with open(f"{outfile_name}.csv", 'r') as manual_obj:
        print(f"Extracting posts from manual input {outfile_name}.csv")
        # use the DictReader to read the csv file
        csv_dict_reader = csv.DictReader(manual_obj)
        # convert the rows to a list
        rows = list(csv_dict_reader)
        # open the output file for writing
        with open(f"out/{outfile_name}Extracted.csv", "w+", newline="\n", encoding="utf-8") as manual_res_file:
            f = csv.writer(manual_res_file)
            # write the headers to the output file
            write_headers(f)
            seen = set()
            # loop through each row in the input file
            for num, row in tqdm(enumerate(rows), total=len(rows)):
                # keep track of a list that checks if the ID has already been seen and if so, skip it
                if row["id"] in seen:
                    print(f"Row with ID '{row['id']}' already seen.")
                    continue
                seen.add(row["id"])
                # get the submission from reddit using the id
                submission = reddit.submission(row["id"])
                # write the relevant information to the output file
                write_fields(f, num + 1, submission, row["comment"])
    print(f"Done extracting posts from manual input {outfile_name}.csv")
    return f"out/{outfile_name}Extracted"


def get_label_and_region(comment, comment_body, i, lr_predictions, mnb_predictions, prev_labels_df, regions_list,
                         rf_predictions, svc_predictions, prev_regions_df, post_region, submission):
    # initialize variables
    prev_data = []
    region = post_region
    label = None
    # check if previous labels dataframe is not None
    if prev_labels_df is not None:
        # get previous data
        prev_data = prev_labels_df[
            (prev_labels_df['ID'] == comment.id) & (prev_labels_df["Title"] == submission.title)
            ]
    if len(prev_data) != 0:
        # get the region from previous data
        region = prev_data.iloc[0]["Region"]
        if region is None or region not in regions_list:
            region = get_region(str(comment.author_flair_text), regions_list, post_region)
        label = prev_data.iloc[0]["Label"]
    else:
        # check if previous regions dataframe is not None
        if prev_regions_df is not None:
            try:
                # get the previous region
                region = prev_regions_df[(prev_regions_df['ID'] == comment.id) & (
                        prev_regions_df["Title"] == submission.title)].iloc[0]["Region"]
            except IndexError:
                # get a new region if there is no previous region
                region = get_region(comment_body, regions_list, post_region)
        else:
            # if there is no previous data, get a new region
            region = get_region(comment_body, regions_list, post_region)
        # if the region is not in the list of regions (so likely an empty string), get the new region
        if region is None or region not in regions_list:
            region = get_region(str(comment.author_flair_text), regions_list, post_region)
        # if the Random Forest classifier is None (and thus all other classifiers except LR), return the LR prediction
        if rf_predictions is None:
            label = lr_predictions[i]
        else:
            # return the first prediction that is not "Unusable." from the classifiers based on score or "Unusable."
            label = next((x for x in [
                lr_predictions[i], svc_predictions[i], rf_predictions[i], mnb_predictions[i]
            ] if x != "Unusable."), "Unusable.")

        # TODO uncomment if one wants to label more comments to gain a larger train dataset for the comments
        # try:
        #     temp_prediction = int(
        #         input(f"\r\nPredicted ''{label}'' correct for?:\r\n {comment_body}\n")
        #     )
        #     if temp_prediction == 1:
        #         label = "Doesn't think ageism exists."
        #     elif temp_prediction == 2:
        #         label = "Acknowledges ageism exists and has a negative association with it."
        #     elif label == 0:
        #         label = "Favors ageism."
        #     else:
        #         label = "Unusable."
        # except ValueError:
        #     pass

        # if the region is not in the list of possible regions, get the fallback region
        if region not in regions_list or region is None:
            region = post_region
    return label, region


def get_classifiers(features, less_labels):
    # Create a logistic regression classifier
    lr_classifier = LogisticRegression(multi_class="multinomial")
    # Create a support vector classifier
    svc_classifier = SVC()
    # Create a random forest classifier
    rf_classifier = RandomForestClassifier()
    # Create a multinomial naive bayes classifier
    mnb_classifier = MultinomialNB()
    # Fit the classifiers to the training data
    mnb_classifier.fit(features, less_labels)
    svc_classifier.fit(features, less_labels)
    rf_classifier.fit(features.toarray(), less_labels)
    lr_classifier.fit(features, less_labels)
    # Return all the classifiers
    return lr_classifier, mnb_classifier, rf_classifier, svc_classifier


def prepare_sklearn_stuff(prev_labels_df):
    # Create a list of tuples for Dataframe rows using list comprehension
    text_and_labels = [tuple(row) for row in [prev_labels_df["Body"], prev_labels_df['Label']]]

    # Extract the texts and labels
    texts_only = list(text_and_labels[0])
    labels = list(text_and_labels[1])
    texts_only, labels = add_short_text_data(texts_only, labels)

    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=preprocess)
    for i, text_only in enumerate(texts_only):
        if not isinstance(text_only, str):
            texts_only[i] = str(text_only)
    features = vectorizer.fit_transform(texts_only)
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=44)

    # Train a Random Forest classifier, Multinomial Naive Bayes classifier, SVM classifier and
    # Logistic Regression classifier
    classifier = RandomForestClassifier()
    classifier.fit(x_train.toarray(), y_train)
    mnb_classifier1 = MultinomialNB()
    mnb_classifier1.fit(x_train, y_train)
    svc_classifier1 = SVC()
    svc_classifier1.fit(x_train, y_train)
    lr_classifier1 = LogisticRegression(multi_class="multinomial")
    lr_classifier1.fit(x_train, y_train)
    lr_classifier2 = LogisticRegression(multi_class="ovr")
    lr_classifier2.fit(x_train, y_train)

    # make predictions
    predictions_test = classifier.predict(x_test)

    # calculate accuracy, precision, recall, f1-score of the classifiers
    accuracy = classifier.score(x_test.toarray(), y_test)
    precision = precision_score(y_test, predictions_test, average='weighted')
    recall = recall_score(y_test, predictions_test, average='weighted')
    f1 = f1_score(y_test, predictions_test, average='weighted')

    # print the results of all classifiers
    print(f"LR score; {lr_classifier1.score(x_test.toarray(), y_test)}, precision; {precision}, recall; {recall}, "
          f"f1; {f1}\nOther accuracies: RF; {accuracy}, SVC; {svc_classifier1.score(x_test.toarray(), y_test)}, "
          f"MNB; {mnb_classifier1.score(x_test.toarray(), y_test)}, "
          f"LR (OvR); {lr_classifier2.score(x_test.toarray(), y_test)}")
    return features, labels, vectorizer


def add_short_text_data(texts_only, labels):
    # add some extra data for short sentences
    texts_only.extend(["Ageism is a serious problem in the tech industry",
                       "Older software engineers have a lot of valuable experience to offer",
                       "It's unfair to discriminate against people based on their age",
                       "Young people are more innovative and bring fresh perspectives to the table",
                       "Ageism is a form of discrimination that needs to be eliminated",
                       "Older workers should be valued and respected, not discriminated against",
                       "Age should not be a factor in hiring and promotion decisions",
                       "Young people are the future of the tech industry",
                       "Experience is more important than age",
                       "Keep learning and you won't experience 'ageism'",
                       "Ageism is a destructive force that holds back progress",
                       "It's important to value and respect people of all ages in software engineering.",
                       "If you're in your 60s you should just quit software engineering.",
                       "Younger developers are better than older ones so these dinosaurs should just retire.",
                       ])
    # also add labels for this data
    labels.extend(["Acknowledges ageism exists and has a negative association with it.",
                   "Acknowledges ageism exists and has a negative association with it.",
                   "Acknowledges ageism exists and has a negative association with it.",
                   "Favors ageism.",
                   "Acknowledges ageism exists and has a negative association with it.",
                   "Acknowledges ageism exists and has a negative association with it.",
                   "Acknowledges ageism exists and has a negative association with it.",
                   "Favors ageism.",
                   "Doesn't think ageism exists.",
                   "Doesn't think ageism exists.",
                   "Acknowledges ageism exists and has a negative association with it.",
                   "Acknowledges ageism exists and has a negative association with it.",
                   "Favors ageism.",
                   "Favors ageism."])
    return texts_only, labels


def get_comments(outfile_name, reddit, train_set_size, num_topics, pickle_folder="auto"):
    # open the csv file with all the relevant Reddit posts
    with open(f"{outfile_name}.csv", 'r') as read_obj:
        csv_dict_reader = csv.DictReader(read_obj)
        print(outfile_name)
        rows = list(csv_dict_reader)
        total_rows = len(rows)
        print(f"Reading comments...")
        # start writing the comments file
        with open(f"{outfile_name}Comments.csv", "w+", newline="\n", encoding="utf-8") as comments_file:
            # install sentiment Natural Language Toolkit package
            try:
                nltk.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.downloader.download('vader_lexicon')
            try:
                nltk.find('sentiment/punkt.zip')
            except LookupError:
                nltk.downloader.download('punkt')

            # install stop word Natural Language Toolkit package
            try:
                nltk.find('corpora/stopwords.zip')
            except LookupError:
                nltk.downloader.download('stopwords')
            try:
                nltk.find('corpora/wordnet.zip')
            except LookupError:
                nltk.downloader.download('wordnet')
            try:
                nltk.find('corpora/omw-1.4.zip')
            except LookupError:
                nltk.downloader.download('omw-1.4')

            # Create a folder to store the labeled data
            if not os.path.exists('./labels/out/'):
                os.makedirs('./labels/out/')
            # Count the number of labeled files
            n = sum(1 for f in os.listdir("./labels/out/") if os.path.isfile(os.path.join("./labels/out/", f)))

            # Read the previous labeled data
            prev_labels_df = None
            prev_regions_df = None
            try:
                prev_labels_df = pd.read_csv(f"./labels/redditManualExtractedLabelled2.csv")
                prev_regions_df = pd.read_csv(f"./labels/out/redditManualExtractedLabelled22.csv")
            except FileNotFoundError:
                print(f"No file: ./labels/{outfile_name}Labelled{n}.csv")

            # write headers
            labels_df = pd.DataFrame(columns=["Number", "Permalink", "Title", "ID", "Body", "Author", "isSubmitter",
                                              "LinkID", "ParentID", "Edited", "Score", "Region", "Label",
                                              "Sentiment1", "Sentiment2", "Age", "Gender"])
            count_df = pd.DataFrame(columns=["ID", "Comment", "Title", "Score", "Comments", "URL", "Domain",
                                             "Permalink", "NoAgeism", "ConfirmAgeism", "FavorAgeism", "Unusable"])
            # initialize AFINN sentiment analysis
            afn = Afinn()
            # initialize the VADER Sentiment Analysis
            sid = SentimentIntensityAnalyzer()
            # initialize the translator
            translator = Translator()

            # get the full list of countries
            regions_list = countries.get_all()

            # initialize classification variables
            features, less_labels, vectorizer = prepare_sklearn_stuff(prev_labels_df)
            lr_classifier, mnb_classifier, rf_classifier, svc_classifier = get_classifiers(features, less_labels)

            # count the number of negative ageism experiences with startups and large companies
            count_startup = 0
            count_large = 0

            # open the csv file that contains all the relevant Reddit posts
            with open(f"./{outfile_name}Computed.csv", "w+", newline="\n", encoding="utf-8") as res_file:
                # iterate over each line of the CSV file as a ordered dictionary
                for row_index, row in tqdm(enumerate(rows), desc="Reading post", total=total_rows, ncols=100,
                                           position=0):
                    # get the submission using the Reddit API
                    submission = reddit.submission(row['ID'])
                    # get all comments from the submission
                    submission.comments.replace_more(limit=None)
                    # remove special characters, URLs, and non-alphanumeric characters from the comments
                    new_comments = [re.sub(r'^>.*\n?|http\S+|[^a-zA-Z0-9 \n.]', '', comment.body, flags=re.MULTILINE)
                                    for comment in submission.comments.list()]
                    # if the submission has no comments
                    if len(new_comments) < 1:
                        print(row["ID"], "has no comments.")
                        continue
                    # get the region of the post
                    post_region = None
                    if "r/cscareerquestionseu/" in row["Permalink"].lower():
                        post_region = "eu"
                    # check if the region is defined in the keyword
                    if "Region: " in row["Keyword"]:
                        post_region = row["Keyword"].replace("Region: ", "")

                    # if the post is gender specific, add this to the region
                    elif " specifically" in row["Keyword"]:
                        post_region = row["Keyword"]

                    # list to store the translated strings
                    elif "Translate: " in row["Keyword"]:
                        # list to store the translated strings
                        translated_comments = []

                        # iterate through the list of strings
                        for string in new_comments:
                            # translate the string
                            translated = translator.translate(string, dest='en')
                            # append the translated string to the list
                            translated_comments.append(translated.text)
                        new_comments = translated_comments
                        post_region = row["Keyword"].replace("Translate: ", "")

                    # get the features of the comments using the TF-IDF vectorizer
                    new_text_features = vectorizer.transform(new_comments)

                    # predict the labels of the comments using the logistic regression classifier
                    lr_predictions = lr_classifier.predict(new_text_features)
                    # initialize variables for other classifier predictions
                    rf_predictions = None
                    svc_predictions = None
                    mnb_predictions = None
                    # TODO uncomment these lines if one wants to use different classifiers
                    # rf_predictions = rf_classifier.predict(new_text_features)
                    # svc_predictions = svc_classifier.predict(new_text_features)
                    # mnb_predictions = mnb_classifier.predict(new_text_features)

                    # iterate over each comment of the submission
                    for i, comment in tqdm(enumerate(submission.comments.list()), desc=f"Comment {row['ID']}",
                                           leave=True, total=len(submission.comments.list()), ncols=100,
                                           position=1):
                        # remove lines that are quoted from other comments
                        comment_body = re.sub(r'^>.*\n?|http\S+|[^a-zA-Z0-9 \n.]', '', comment.body, flags=re.MULTILINE)
                        # write the comments CSV file if the comment is not deleted or removed
                        if comment_body != "[deleted]" and comment_body != "[removed]":
                            # get the label and region
                            label, region = get_label_and_region(comment, comment_body, i, lr_predictions,
                                                                 mnb_predictions, prev_labels_df, regions_list,
                                                                 rf_predictions, svc_predictions, prev_regions_df,
                                                                 post_region, submission)
                            age = None
                            # get the age from the comment
                            numbers = re.findall(r'\b[0-9]\w+', comment_body)
                            if len(re.findall(r'\b[0-9]\w+', comment_body)) > 0:
                                for age_num in numbers:
                                    num = age_num.lower()
                                    # if the age ends with YO or AGEsomething, return that
                                    if num.endswith("yo") or "some" in num or "year" in num or \
                                            num.endswith("years") or num.endswith("early") or "old" in num or \
                                            num.endswith("yr") or num.endswith("yrs"):
                                        age = num
                                if age is None:
                                    # otherwise, return the first number
                                    age = re.findall(r'\b[0-9]\w+', comment_body)[0]
                            elif len(re.findall(r'\b[0-9]\w+', submission.title)) > 0:
                                age = re.findall(r'\b[0-9]\w+', submission.title)[0]

                            #  check for gender (unfortunately not working properly on the data)
                            user_gender = "-"
                            if comment_body in ["masculine", " men", " male", "gentleman", "boy", "guy", " son ",
                                                " lad ", "dude", "bro", "father", " man "]:
                                user_gender = "Male"
                            elif comment_body in ["wife", "women", "woman", "femin", "lady", "girl", "sister",
                                                  "gentlewoman", "mother"]:
                                user_gender = "Female"

                            # check for startup and large company
                            if label == 'Acknowledges ageism exists and has a negative association with it.':
                                if 'startup' in comment_body.lower() or 'start up' in comment_body.lower():
                                    count_startup += 1
                                if 'large compan' in comment_body.lower() or 'big compan' in comment_body.lower() \
                                        or 'large firm' in comment_body.lower() or 'big firm' in comment_body.lower():
                                    count_large += 1

                            # add the list of comments to the dataframe
                            labels_df = labels_df.append(
                                {"Number": row["Number"], "Permalink": row["Permalink"], "Title": row["Title"],
                                 "ID": comment.id, "Body": comment_body, "Author": comment.author,
                                 "isSubmitter": comment.is_submitter, "LinkID": comment.link_id,
                                 "ParentID": comment.parent_id, "Edited": comment.edited,
                                 "Score": comment.score, "Region": region, "Label": label,
                                 "Sentiment1": sid.polarity_scores(comment_body)['compound'],
                                 "Sentiment2": afn.score(comment_body), "Age": age, "Gender": user_gender},
                                ignore_index=True
                            )
                    # keep track of some stats per post
                    count_df = count_df.append(
                        {"ID": submission.id, "Comment": row["Keyword"], "Title": submission.title,
                         "Score": submission.score, "Comments": submission.num_comments,
                         "URL": submission.url, "Domain": submission.domain,
                         "Permalink": row["Permalink"],
                         "NoAgeism": labels_df[
                             labels_df['Label'] == "Doesn't think ageism exists."].groupby(
                             ['Permalink']).size().get(row["Permalink"], 0),
                         "ConfirmAgeism": labels_df[labels_df[
                                                        'Label'] == "Acknowledges ageism exists "
                                                                    "and has a negative association with it."].groupby(
                             ['Permalink']).size().get(row["Permalink"], 0),
                         "FavorAgeism": labels_df[
                             labels_df['Label'] == "Favors ageism."].groupby(
                             ['Permalink']).size().get(row["Permalink"], 0),
                         "Unusable": labels_df[labels_df['Label'] == "Unusable."].groupby(
                             ['Permalink']).size().get(row["Permalink"], 0),
                         "NoAgeismUp": labels_df[
                             labels_df['Label'] == "Doesn't think ageism exists."].groupby(
                             ['Permalink'])['Score'].sum().get(row["Permalink"], 0),
                         "ConfirmAgeismUp": labels_df[
                             labels_df['Label'] == "Acknowledges ageism exists and has a "
                                                   "negative association with it."].groupby(
                             ['Permalink'])['Score'].sum().get(row["Permalink"], 0),
                         "FavorAgeismUp": labels_df[
                             labels_df['Label'] == "Favors ageism."].groupby(
                             ['Permalink'])['Score'].sum().get(row["Permalink"], 0),
                         "UnusableUp": labels_df[labels_df['Label'] == "Unusable."].groupby(
                             ['Permalink'])['Score'].sum().get(row["Permalink"], 0),
                         "NoAgeismSent1": labels_df[
                             labels_df['Label'] == "Doesn't think ageism exists."].groupby(
                             ['Permalink'])['Sentiment1'].mean().get(row["Permalink"], 0),
                         "ConfirmAgeismSent1": labels_df[
                             labels_df['Label'] == "Acknowledges ageism exists and has a "
                                                   "negative association with it."].groupby(
                             ['Permalink'])['Sentiment1'].mean().get(row["Permalink"], 0),
                         "FavorAgeismSent1": labels_df[
                             labels_df['Label'] == "Favors ageism."].groupby(
                             ['Permalink'])['Sentiment1'].mean().get(row["Permalink"], 0),
                         "NoAgeismSent2": labels_df[
                             labels_df['Label'] == "Doesn't think ageism exists."].groupby(
                             ['Permalink'])['Sentiment2'].mean().get(row["Permalink"], 0),
                         "ConfirmAgeismSent2": labels_df[
                             labels_df['Label'] == "Acknowledges ageism exists and has a "
                                                   "negative association with it."].groupby(
                             ['Permalink'])['Sentiment2'].mean().get(row["Permalink"], 0),
                         "FavorAgeismAge": labels_df[
                             labels_df['Label'] == "Favors ageism."].groupby(
                             ['Permalink'])['Sentiment2'].mean().get(row["Permalink"], 0),
                         },
                        ignore_index=True)
                    # get the sentiment analyses data
                    grouped_df = labels_df.groupby(["Region", "Label"]).agg({
                        "ID": "count",
                        "Score": ["sum", "mean"],
                        "Sentiment1": ["sum", "mean", "std"],
                        "Sentiment2": ["sum", "mean", "std"],
                    })
                    # get the age data
                    grouped_df2 = labels_df.groupby(["Age", "Label"]).agg({
                        "ID": "count",
                        "Score": ["sum", "mean"],
                        "Sentiment1": ["sum", "mean", "std"],
                        "Sentiment2": ["sum", "mean", "std"],
                    })
                    # get the gender data
                    grouped_df3 = labels_df.groupby(["Gender", "Label"]).agg({
                        "ID": "count",
                        "Score": ["sum", "mean"],
                        "Sentiment1": ["sum", "mean", "std"],
                        "Sentiment2": ["sum", "mean", "std"],
                    })

                # write the grouped dataframes from above in CSV files
                with open(f"./{outfile_name}Diversity.csv", "w+", newline="\n", encoding="utf-8") as div_file:
                    grouped_df.to_csv(div_file)
                with open(f"./{outfile_name}Age.csv", "w+", newline="\n", encoding="utf-8") as div_file:
                    grouped_df2.to_csv(div_file)
                with open(f"./{outfile_name}Gender.csv", "w+", newline="\n", encoding="utf-8") as div_file:
                    grouped_df3.to_csv(div_file)

                # write the comments "NoAgeism", "ConfirmAgeism", "FavorAgeism", "Unusable"
                total_no_ageism = count_df['NoAgeism'].sum()
                total_confirm_ageism = count_df['ConfirmAgeism'].sum()
                total_favor_ageism = count_df['FavorAgeism'].sum()
                total_unusable = count_df['Unusable'].sum()
                total_comments = total_favor_ageism + total_no_ageism + total_confirm_ageism
                # print stats on the number of posts
                print(f"\nNo ageism: {total_no_ageism} ({total_no_ageism / total_comments})\n"
                      f"Confirm ageism: {total_confirm_ageism} ({total_confirm_ageism / total_comments})\n"
                      f"Favor ageism: {total_favor_ageism} ({total_favor_ageism / total_comments})\n"
                      f"Unusable: {total_unusable} ({total_unusable / total_comments})\n"
                      f"Negative startups: {count_startup} ({count_startup / total_confirm_ageism})\n"
                      f"Negative large companies: {count_large} ({count_large / total_confirm_ageism})\n"
                      f"total: {total_no_ageism + total_confirm_ageism + total_favor_ageism + total_unusable} -"
                      f" {total_comments}")
                total_no_ageism_score = count_df['NoAgeismUp'].sum()
                total_confirm_ageism_score = count_df['ConfirmAgeismUp'].sum()
                total_favor_ageism_score = count_df['FavorAgeismUp'].sum()
                total_unusable_score = count_df['UnusableUp'].sum()
                total_score = total_favor_ageism_score + total_confirm_ageism_score + total_no_ageism_score
                # print stats on the number of likes
                print(f"\nVOTES/SCORE/LIKES\n"
                      f"No ageism: {total_no_ageism_score} ({total_no_ageism_score / total_score})\n"
                      f"Confirm ageism: {total_confirm_ageism_score} ({total_confirm_ageism_score / total_score})\n"
                      f"Favor ageism: {total_favor_ageism_score} ({total_favor_ageism_score / total_score})\n"
                      f"Unusable: {total_unusable_score} (a lot)\n"
                      f"total score {total_score}, comments {total_comments}")
                count_df.to_csv(res_file)
            # write the new labels into a csv file
            labels_df.to_csv(f"./labels/{outfile_name}Labelled{n + 1}.csv")
            print(type(count_df.iloc[0]['NoAgeism']))
            print(f"Wrote categories to ./labels/{outfile_name}Labelled{n + 1}.csv")

    print("Done reading comments")
