## import necessary modules
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics

# for keyword extraction
from rake_nltk import Rake

from tqdm.notebook import tqdm
# Initialize TQDM
tqdm.pandas(desc="progress bar")
from tqdm import trange

from textblob import TextBlob
import nltk
# nltk.download("all")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import umap.umap_ as umap

import re

# for clustering
import hdbscan
from sklearn.metrics import silhouette_score
# from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.metrics.cluster import adjusted_mutual_info_score

# for visualizations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
rake_nltk_var = Rake()


stop_words = list(set(stopwords.words("english")))
stop_words += ["jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june", "jul", "july", "aug", "august", "sep", "september", "oct", "october", "nov", "november", "dec", "december"]


## udf to remove stop words
def remove_stop_words(text):
	tokenized_text = word_tokenize(text)
	final_text_ls = []
	for i in tokenized_text:
		if i not in stop_words:
			final_text_ls.append(i)
	return " ".join([str(i) for i in final_text_ls])

## preprocess the string with removing stop words and punctuations
def process_strings(text):
	## convert all strings to lower case
	text = text.lower()

	## remove numbers and stop words from the text - this messes up the keyword extraction
	## removing all punctuations and numerical characters
	s1 = re.sub('[0-9]+', '', text)  # removing only the numerical values

	# remove unnecessary white spaces
	s2 = re.sub(' +', ' ', s1)
	s3 = re.sub(' ,+', ',', s2)

	# remove stop words
	s4 = remove_stop_words(s3)
	return s4


def return_keywords_rake(text, top_n=5):
	rake_nltk_var.extract_keywords_from_text(text)
	return rake_nltk_var.get_ranked_phrases()[:top_n]