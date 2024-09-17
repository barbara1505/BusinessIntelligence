# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import KMeans


def load_and_preprocess_data(dataset_path):
    data = pd.read_csv(dataset_path, encoding='latin')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 576)

    # General information about the dataset
    print(data.info())
    print(data.shape)
    print(data.head())

    # Check for missing values
    # miss_val = data.isnull().sum
    # print(miss_val)

    # Remove rows with null values in 'CustomerID' and 'Description'
    data = data.dropna(subset=['CustomerID', 'Description'])

    # Filter rows where 'Quantity' is non-negative and 'UnitPrice' is >0
    data = data.loc[(data.Quantity >= 0) & (data.UnitPrice > 0)]

    # Dataset after cleaning
    print(data.info())
    print(data.shape)
    print(data.describe())

    return data


def plot_RFM_values_distribution(RFM_data):
    grid = plt.figure(figsize=(24, 12))
    grid.add_subplot(221)
    sns.regplot(x='Recency', y='Monetary', data=RFM_data)
    grid.add_subplot(222)
    sns.regplot(x='Frequency', y='Monetary', data=RFM_data)
    grid.add_subplot(223)
    sns.regplot(x='Recency_Normalized', y='Monetary_Normalized', data=RFM_data)
    grid.add_subplot(224)
    sns.regplot(x='Frequency_Normalized', y='Monetary_Normalized', data=RFM_data)
    plt.show()

    RFM_3d = plt.figure(figsize=(12, 10))
    ax = RFM_3d.add_subplot(111, projection='3d')
    xs = RFM_data.Recency_Normalized
    ys = RFM_data.Frequency_Normalized
    zs = RFM_data.Monetary_Normalized
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.scatter(xs, ys, zs, s=5)
    plt.show()


def plot_elbow_kmeans(criteria, criteria_params):
    plt.figure(figsize=(24, 12))
    plt.plot(range(1, 10), criteria_params, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel(f'{criteria}')
    plt.title(f'The Elbow Method using {criteria}')
    plt.show()


def customer_segmentation(dataset_path):
    # Load&Preprocess data
    data = load_and_preprocess_data(dataset_path)

    # RFM model
    data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate, format='%m/%d/%y %H:%M')
    most_recent_date = data.InvoiceDate.max()
    reference_date = most_recent_date + pd.Timedelta(days=1)

    print('Most recent date:', most_recent_date)
    print('Reference date:', reference_date)

    # Calculating Recency
    data['Recency'] = (reference_date - data.InvoiceDate).dt.days
    # print(data['Recency'])

    RFM_data = pd.DataFrame(data[['CustomerID', 'Recency']])
    # Calculating the minimum value of Recency for each customer.
    RFM_data = RFM_data.groupby('CustomerID').min().reset_index()

    print(RFM_data['Recency'].head())
    # print(data['CustomerID'].unique().shape)

    # Calculate Total amount of money spent in each transaction
    data['TotalPrice'] = data.Quantity * data.UnitPrice

    # Calculating Frequency
    customer_frequency = data.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    customer_frequency.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)
    print(customer_frequency.head(20))

    RFM_data['Frequency'] = customer_frequency['Frequency']

    # Calculating Monetary
    customer_monetary = data[['CustomerID', 'TotalPrice']].groupby(['CustomerID']).sum().reset_index()
    RFM_data['Monetary'] = customer_monetary['TotalPrice']
    print(customer_monetary.head())

    # Write results into csv file
    RFM_data.to_csv("online_retail_RFM.csv")

    # Customer segmentation
    RFM_model = pd.read_csv("online_retail_RFM.csv", encoding='latin')
    pd.set_option('display.max_columns', None)
    print(RFM_model.describe())

    # Logarithmic transformation - calculate log(x+1) to reduce skewness in the data.
    RFM_model['Recency_Normalized'] = RFM_model.Recency.apply(math.log1p)
    RFM_model['Frequency_Normalized'] = RFM_model.Frequency.apply(math.log1p)
    RFM_model['Monetary_Normalized'] = RFM_model.Monetary.apply(math.log1p)

    plot_RFM_values_distribution(RFM_model)

    # Standardization

    RFM_normalized = RFM_model[['Recency_Normalized', 'Frequency_Normalized', 'Monetary_Normalized']]
    standard_scaler = preprocessing.StandardScaler().fit(RFM_normalized)
    RFM_scaled = standard_scaler.transform(RFM_normalized)
    print(RFM_scaled)

    RFM_scaled = pd.DataFrame(RFM_scaled, columns=['Recency', 'Frequency', 'Monetary'])  # Renaming the columns
    # Elbow method to determine optimal number of clusters for KMeans
    inertias = []
    distortions = []
    for k in range(1, 10):
        KMeans_model = KMeans(n_clusters=k).fit(RFM_scaled)
        inertias.append(KMeans_model.inertia_)
        distortions.append(
             sum(np.min(cdist(RFM_scaled, KMeans_model.cluster_centers_, 'euclidean'), axis=1)) / RFM_scaled.shape[0])

    plot_elbow_kmeans("Inertia", inertias)
    # plot_elbow_kmeans("Distortion", distortions)

    # Optimal number of clusters is 4.
    n_clusters_elbow = 4
    k_means = KMeans(n_clusters=n_clusters_elbow, random_state=42)
    k_means.fit(RFM_scaled)

    # Create a new DataFrame with CustomerID, Recency, Frequency, Monetary, and Cluster
    RFM_clustered = pd.DataFrame({
        'CustomerID': RFM_model['CustomerID'],
        'Recency': RFM_model['Recency'],
        'Frequency': RFM_model['Frequency'],
        'Monetary': RFM_model['Monetary'],
        'Cluster': k_means.labels_
    })
    print(RFM_clustered.head())

    # Centroids
    centroids = RFM_clustered.groupby('Cluster')[['CustomerID', 'Recency', 'Frequency', 'Monetary']].agg({
        'CustomerID': 'nunique',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    centroids = centroids.rename(columns={'CustomerID': 'NumCustomers'})
    print(centroids)

    RFM_clustered = RFM_clustered.replace({
        'Cluster': {0: 'NewCustomer', 1: 'Loyal', 2: 'CantLoseThem', 3: 'AtRisk'}
    })
    RFM_clustered.to_csv("online_retail_clustered.csv")


if __name__ == '__main__':
    customer_segmentation("online_retail.csv")
