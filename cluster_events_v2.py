import sys

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler


def main_v2(features_filepath, labels_filepath, cluster_cutoff, label_cutoff):
    features = np.loadtxt(features_filepath, delimiter=",")
    labels = np.loadtxt(labels_filepath, delimiter=",")
    print(features.shape)
    print()

    # scale features
    ss = StandardScaler()
    features_scaled = ss.fit_transform(features)

    dbscan = DBSCAN(eps=.6, min_samples=8)
    agglo = AgglomerativeClustering(n_clusters=16)
    kmeans = KMeans(n_clusters=16)
    meanshift = MeanShift(n_jobs=-1)

    for model, name in ((agglo, "Agglomerative"), (kmeans, "K-Means")): #(meanshift, "Mean Shift")):
        print(name)

        model.fit(features_scaled)

        clusters = model.labels_
        cluster_names, cl_counts = np.unique(clusters, return_counts=True)
        label_names, l_counts = np.unique(labels, return_counts=True)

        label_names = list(label_names)  # so I can use .index() in the frequency calculations

        print("CLUSTERS:")
        print(cluster_names)
        print(cl_counts)

        print("LABELS:")
        print(label_names)
        print(l_counts)

        # make df of events
        #   (ix, cluster, label)
        df = pd.DataFrame(list(zip(clusters, labels)), columns=["Cluster", "Label"])
        df["Cluster_Freq"] = 0.0
        df["Label_Freq"] = 0.0
        df["Is_Outlier"] = 0
        df["Outlier_Gold"] = 0

        offset = 0
        if -1 in clusters:
            offset = 1

        # calculate cluster frequencies and label frequencies
        # get all events deemed to be outliers (in a class representing <X% of total)
        for i, r in df.iterrows():
            df.at[i, "Cluster_Freq"] = cl_counts[int(r["Cluster"])+offset] / len(df)
            df.at[i, "Label_Freq"] = l_counts[label_names.index(int(r["Label"]))] / len(df)

            if df.at[i, "Cluster_Freq"] < cluster_cutoff:
                df.at[i, "Is_Outlier"] = 1
            if df.at[i, "Label_Freq"] < label_cutoff:
                df.at[i, "Outlier_Gold"] = 1

        outliers_df = df.loc[df.Is_Outlier == 1]
        precision = metrics.precision_score(outliers_df.Outlier_Gold, outliers_df.Is_Outlier)
        print("Precision: ", precision)

        # determine tp/fp/fn/tn?

        # calculate metrics

        print()


if __name__ == "__main__":
    main_v2(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
