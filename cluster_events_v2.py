import sys

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import DBSCAN


def main(features_filepath, features_raw_filepath):
    features = np.loadtxt(features_filepath, delimiter=",")
    features_raw = None
    if features_raw_filepath:
        features_raw = np.loadtxt(features_raw_filepath, delimiter=",")
    print(features.shape)

    labels = features[:, -1]
    print(labels)
    features = features[:, :-1]
    print(features.shape)

    dbscan = DBSCAN(eps=0.7, min_samples=24).fit(features)
    clusters = dbscan.labels_

    outliers = np.where(clusters == -1)[0]  # np.where() returns a length-1 tuple where only the first value is the actual output??? For some reason???

    cluster_names, cl_counts = np.unique(clusters, return_counts=True)
    print("CLUSTERS:")
    print(cluster_names)
    print(cl_counts)

    label_ids, label_counts = np.unique(labels, return_counts=True)

    print("LABELS:")
    print(label_ids)
    print(label_counts)

    f1 = 3  # teal intercept
    f2 = 11  # red intercept
    feature1 = features_raw[:, f1]
    feature2 = features_raw[:, f2]

    # plot clusters

    # plot labels

    matrix = metrics.cluster.contingency_matrix(labels, clusters)
    df = pd.DataFrame(matrix, index=label_ids, columns=cluster_names)
    print(df)
    print("Normalized mutual info score:")
    print(metrics.normalized_mutual_info_score(labels, clusters))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
