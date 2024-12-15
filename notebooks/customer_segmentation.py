# Customer segmentation

# k-means clustering


import pandas as pd

# load the dataset
file_path = "Mall_Customers.csv"
df = pd.read_csv(file_path)

# preview the dataset
print(df.head())

# basic information about the dataset
print(df.info())
print(df.describe())




# check for missing values
missing_values = df.isnull().sum()

print("Missing values in each column: ")
print(missing_values)




# exploratory data analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

# distribution of Age
sns.histplot(df['Age'], kde = True, bins = 15, color = 'blue')
plt.title('Age Distribution')
plt.show()

# Spending Score vs. Annual Income
plt.figure(figsize = (8,6))
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', data = df)
plt.title('Spending Score vs. Annual Income')




from sklearn.preprocessing import StandardScaler

# selecting relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# k-means clustering

from sklearn.cluster import KMeans
import numpy as np

# determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# plot the elbow curve
plt.figure(figsize = (8, 6))
plt.plot(k_values, inertia, marker = 'o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show




# Silhouette Score

from sklearn.metrics import silhouette_score

# range of k values to evaluate
k_values = range(2, 11)
silhouette_scores = []

# calculating silhouette scores for each k
for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    cluster_labels = kmeans.labels_
    score = silhouette_score(X, cluster_labels)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker = 'o')
plt.title('Silhouette Scores for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid()
plt.show()

# find the optimal k
optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f'The optimal number of clusters based on silhouette score is: {optimal_k}')




# fitting the model with the optimal number of clusters 
kmeans = KMeans(n_clusters = 5, random_state = 42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# visualize the clusters
plt.figure(figsize = (8, 6))
sns.scatterplot(
    x = 'Annual Income (k$)', y = 'Spending Score (1-100)',
    hue = 'Cluster', data = df, palette = 'viridis'
)
plt.title('Customer Segments')
plt.show()




# analyze clusters

# group by cluster and calculate mean values
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary)

# count of customers per cluster
print(df['Cluster'].value_counts())

