#!/usr/bin/env python

## import necessary modules
import pandas as pd
import numpy as np
import pickle

# for generating BERT embeddings
from sentence_transformers import SentenceTransformer

# for dimensionality reduction and visualization
import umap.umap_ as umap

# for clustering
import hdbscan

# for visualization
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# for getting the feature relevance while extracting top 10 features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# for getting all the helper functions
import helper_functions as hf

import warnings
warnings.filterwarnings('ignore')


## reading the dataset
try:
	print("Reading the input file")
	df_raw = pd.read_csv("input.csv")
	df = df_raw.copy(deep=True) # creating a copy of the original dataframe in case something gets messed up, wouldn't read the file multiple times
	print("Input file read successfully")
except:
	print("Error in reading file... Please check file location specified. \nAborting execution\n")

## preprocessing reviews
try:
	print("\nPreprocessing the review strings...")
	## preprocessing the review contents
	df['processed_content'] = df['content'].apply(lambda x: hf.process_strings(x))
	print("Preprocessing completed")
except:
	print("Error in preprocessing... \nAborting execution\n")

## extracting keyphrases
try:
	print("\nExtracting keywords and keyphrases...")
	df['keywords'] = df['processed_content'].apply(lambda x: hf.return_keywords_rake(x, top_n=5))
	print("Keyword extraction sucessfull")
except:
	print("Error in keyword extraction... Aborting execution\n")

## creating the exploded view of the ids and the respective keywords
ls_for_df = []

for id in df.id.unique():
    for i, keyword in enumerate(df[df['id'] == id]['keywords'].iloc[0]):
        ls_for_df.append([str(id) + "_" + str(i+1), keyword])


df_with_id_keywords = pd.DataFrame(ls_for_df, columns = ['modified_id', 'keyword'])


## removing duplicated keywords (if any)
df_with_id_keywords['deduplication_string'] = df_with_id_keywords.apply(lambda x: x['modified_id'].split("_")[0] + ":" + x['keyword'], axis=1)
df_with_id_keywords['deduplication_flag'] = df_with_id_keywords['deduplication_string'].duplicated()
## removing all the duplicated rows
df_with_id_keywords = df_with_id_keywords[df_with_id_keywords['deduplication_flag'] == False]
df_with_id_keywords.drop(columns = ['deduplication_flag', 'deduplication_string'], inplace=True)


## creating the corpus of words before embedding
corpus = list(df_with_id_keywords['keyword'])
print(f"The length of the corpus is {len(corpus)}")


## create the embeddings
# print("\nCreating BERT embeddings for all the keywords")
# embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
# corpus_embeddings = embedder.encode(corpus)

# # saving the corpus embedding as a pickle file
# print("Saving the BERT embeddings")
# with open('preprocessed/embeddings/keyword_embeddings.pickle', 'wb') as pkl:
#     pickle.dump(corpus_embeddings, pkl)

# reading the pickle file from the already generated embeddings
print("\nReading the existing bert embeddings")
with open('preprocessed/embeddings/keyword_embeddings.pickle', 'rb') as pkl:
    corpus_embeddings = pickle.load(pkl)

print(f"Shape of individual embedding: {len(corpus_embeddings[0])}")

## reducing dimensions using UMAP
try:
	print("\nReducing Dimensions using UMAP...")
	clusterable_embedding = umap.UMAP(
	    n_neighbors=40, # this should be kept large, because we want more number of values to come into a cluster
	    min_dist=0.0, # this should be kept small as we want more tightly packed densities to be together
	    n_components=2, #reducing dimension to 2
	    random_state=42,
	).fit_transform(corpus_embeddings)
	print("Dimensionality reduction successful")
except:
	print("Error in Dimensionality Reduction... \nAborting execution\n")

## generating clusters
try:
	print("\nGenerating Clusters...")
	labels = hdbscan.HDBSCAN(
	    min_samples=50,
	    min_cluster_size=210,
	).fit_predict(clusterable_embedding)
	print(f"Generated {len(set(labels))} clusters")
except:
	print("Error in generating clusters... \nAborting execution\n")

print("Generating visualization for the clusters generated")

plt.figure(figsize=(20, 15))
plt.title("Clusters")
plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=labels, s=0.1, cmap='Spectral');
plt.savefig('preprocessed/figures/cluster_vizualization.png', bbox_inches='tight')

print("\nCluster visualization saved")

## generating dataframe with the keywords and the respective cluster labels
df_cluster = df_with_id_keywords.copy(deep=True)
assert len(labels) == df_cluster.keyword.shape[0]
df_cluster['cluster'] = labels

# getting all the texts present in each of hte clusters
cluster_texts = {}
for cluster in range(-1, max(df_cluster.cluster.unique())+1):
    cluster_texts[str(cluster)] = "; ".join(text for text in df_cluster[df_cluster['cluster'] == cluster]['keyword'])

print("\nGenerating worclouds for each of the clusters to find the most predominant words in each of the clusters")

for cluster in range(-1, max(df_cluster.cluster.unique())+1):
    #print(f"Generating Word Cloud for cluster {cluster}")
    # Creating word_cloud with text as argument 
    wc = WordCloud(collocations = False, background_color = 'white').generate(cluster_texts[str(cluster)])

    # Display the generated Word Cloud
    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f"For Cluster: {cluster}")
    plt.savefig(f"preprocessed/figures/wordclouds/cluster_{str(cluster)}_wordcloud.png", bbox_inches='tight')
print("All wordcloud images saved")


## creating manual cluster mappings
cluster_mapping = {-1 : 'taste', 
0 : 'appearance', 
1 : 'alcohol', 
2 : 'ingredients', 
3 : 'style', 
4 : 'ingredients', 
5 : 'carbonation', 
6 : 'price', 
7 : 'style', 
8 : 'style', 
9 : 'palate', 
10 : 'taste', 
11 : 'place_of_service', 
12 : 'packaging', 
13 : 'packaging', 
14: 'noise', 
15 : 'style', 
16 : 'style', 
17: 'season', 
18 : 'taste', 
19 : 'ingredients', 
20 : 'ingredients', 
21 : 'taste', 
22 : 'style', 
23 : 'taste', 
24 : 'finish', 
25 : 'style', 
26 : 'style', 
27 : 'ingredients', 
28 : 'ingredients', 
29 : 'taste', 
30 : 'ingredients', 
31 : 'ingredients', 
32 : 'aroma', 
33 : 'aroma', 
34 : 'aroma', 
35 : 'taste',
36 : 'ingredients'}

## renaming the clusters according to the mapping generated
df_cluster.replace({'cluster': cluster_mapping}, inplace=True)

## removing the clusters which had been tagged as "noise" - basically where the information was random 
df_cluster2 = df_cluster[df_cluster['cluster'] != 'noise']

df_cluster2['modified_id'] = df_cluster2['modified_id'].apply(lambda x: x.split("_")[0])


## saving this file for use later on
df_cluster2.to_csv("preprocessed/review_attribute_exploded_tagging.csv", index=False)

## grouping by the 'modified_ids' and getting all the relevant attributes
print("\nGrouping all the attributes by the ids")
id_attribute_mapping = df_cluster2.groupby(['modified_id']).agg({'cluster': 'unique'}).reset_index()
id_attribute_mapping['attribute'] = id_attribute_mapping['cluster'].apply(lambda x: ", ".join(str(x)))


id_attribute_mapping.drop(columns=['cluster'], inplace=True)
id_attribute_mapping.rename(columns={'modified_id':'id'}, inplace=True)

print("\nSaving review attribute mapping file")
try:
	id_attribute_mapping.sort_values('id').to_csv("sc26040_attributes.csv", index=False)
	print("File saved successfully")
except:
	print("Error in saving file")


#### top 10 attributes ####


# df_with_attributes = pd.read_csv("outputs/preprocessed/review_attribute_exploded_tagging.csv")
df_with_attributes = df_cluster2.copy(deep=True) # use the existing dataframe in case the preprocessed file is not being delivered.

# one-hot encoding the different attributed
print("\nOne-hot encoding the features/attributes")
try:
	one_hot_encoded_data = pd.get_dummies(df_with_attributes, columns = ['cluster'])
	print("Completed one-hot encoding")
except:
	print("Error in one-hot encoding... \n Aborting execution\n")

# dropping the keywords column
one_hot_encoded_data.drop(columns = ['keyword'], inplace=True)

# generating a dictionary which will be used for aggregation of all the attributes (features) by their sum
one_hot_encoded_data.columns.unique()
cluster_agg_dict = {}
for column in one_hot_encoded_data.columns.unique():
    if('cluster_' in column):
        cluster_agg_dict[column] = 'sum'

# doing the aggregation because, in case for an attribute there's the mention of an attribute multiple times, it would get added up and contain more weight than the other features
one_hot_encoded_data_grouped = one_hot_encoded_data.groupby(['modified_id']).agg(cluster_agg_dict).reset_index()

one_hot_encoded_data_grouped['modified_id'] = one_hot_encoded_data_grouped['modified_id'].astype('int64')
df['modified_id'] = df['id'].astype('int64')
merged_df = one_hot_encoded_data_grouped.merge(df[['id', 'rating']], left_on='modified_id', right_on='id', how='inner')
merged_df.drop(columns = ['modified_id'], inplace=True)

## using random forest to select the features with the highest feature scores as the top features
## the idea is to see which features are the ones which are responsible for giving the target - rating in this case

features = [i for i in one_hot_encoded_data.columns.unique() if 'cluster_' in i]

X = merged_df[features].to_numpy()
y = merged_df['rating'].to_numpy()
## converting the y values to float and rounding them up to 2 places of decimal
y = [float(x) for x in y]
y = np.round(y, 2)

## running a gridsearch to find the best parameters
## define the grid of parameters
param_grid = { 
    'n_estimators': [300, 400, 500, 600, 700, 800, 900, 1000, 1100],
    'max_depth' : [4,5,6,7]
}

## instantiating a separate model for running the gridsearch parameters
rfr = RandomForestRegressor(random_state=1997)

print("\nRunning Gridsearch to find the best n_estimators and max_depth...")
# CV_rfr = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5)
# CV_rfr.fit(X, y)

## getting the best parameters
# best_parameters = CV_rfr.best_params_
best_parameters = {'max_depth': 6, 'n_estimators': 700} ## using the answers from previous run because GridSearch takes a long time
print(f"Best Parameters obtained: {best_parameters}")


## initiating the random forest classifier
rnd_reg = RandomForestRegressor(n_estimators=best_parameters['n_estimators'],
								max_depth=best_parameters['max_depth'], 
								max_features='auto', 
								n_jobs=-1, 
								random_state=1997)

features = [i for i in one_hot_encoded_data.columns.unique() if 'cluster_' in i]

X = merged_df[features].to_numpy()
y = merged_df['rating'].to_numpy()

## fitting
rnd_reg.fit(X, y)

## getting the feature scores
Feature_score = zip(features, rnd_reg.feature_importances_)

df_ls = []
for name, score in Feature_score:
    df_ls.append([name, score*100])

## creating a dataframe for ease of manipulation   
importance_df = pd.DataFrame(df_ls, columns = ['feature', 'score'])
importance_df['feature'] = importance_df['feature'].apply(lambda x: x.replace("cluster_", ""))
importance_df.sort_values('score', ascending=False, inplace=True)

importance_df.rename(columns = {'feature': 'top10'}, inplace=True)

# saving top 10 attributes for second submission file
print("\nSaving top 10 attributes")
try:
	importance_df['top10'].head(10).to_csv("sc26040_top10.csv", index=False)
	print("File saved successfully")
except:
	print("Error in saving file")