
# NOTE: THIS SCRIPT IS IN THE PROCESS OF BECOMING DEPRECATED

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.cluster import DBSCAN


SAMPLE_FILEPATH = "notebooks/mockdata.npz"
FULL_FILEPATH = "notebooks/fulldata.npz"
LONGCURVES_FILEPATH = "notebooks/fulldata_longcurves.npz"
NUM_PASSBANDS = 6

CLASS_NAMES = {
    0: "SNI-Combined",
    15: "TDE",
    42: "SNII",
    52: "SNIax",
    62: "SNIbc",
    67: "SNIa-91bg",
    88: "AGN",
    90: "SNIa",
    95: "SLSN-I",
    992: "ILOT",
    993: "CaRT",
    994: "PISN"
}


def plot_event(params, event_id, times, fluxes, filters, plot_regression=True):
    num_times = len(times[event_id])
    time_row = times[event_id]
    flux_row = fluxes[event_id]
    filter_row = filters[event_id]

    for passband, color in zip(range(NUM_PASSBANDS), ("blue", "cyan", "green", "yellow", "orange", "red")):
        good_ixes = np.where(filter_row==str(passband))
        plt.plot(time_row[good_ixes], flux_row[good_ixes], color=color, marker="o")

        PPP = int(len(params)/NUM_PASSBANDS)  # params per passband
        points = np.zeros((num_times, 2))

        slope, intercept = params[passband * PPP:(passband + 1) * PPP]

        # generate points
        if plot_regression:
            for i, t in enumerate(times[event_id]):
                f = slope * t + intercept
                points[i] = [t, f]

            plt.plot(points[:, 0], points[:, 1], color=color, linestyle="dashed")

    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.ylim(-8.0, 0.1)
    if plot_regression:
        plt.title("Intermediate Regression Curve {}".format(event_id))
    else:
        plt.title("Light Curve {}".format(event_id))
    plt.show()


def get_features(ix, times, fluxes, filters):
    e_fluxes = [[] for i in range(NUM_PASSBANDS)]

    for passband, time, flux in zip(filters[ix], times[ix], fluxes[ix]):
        e_fluxes[int(passband)].append((time, flux))

    features = [0 for i in range(NUM_PASSBANDS * 2)]

    for passband in range(NUM_PASSBANDS):
        if len(e_fluxes[passband]) >= 1:
            X = np.array(e_fluxes[passband])

            linreg = LinearRegression()
            linreg.fit(X[:, 0:1].astype(np.float32), X[:, 1:2])

            features[passband * 2] = linreg.coef_[0][0]
            features[1 + passband * 2] = linreg.intercept_[0]

    return features


def main(model, filepath, to_analyze=-1, plot_all=False, plot_outliers=False,
         plot_classes=False):
    data = np.load(filepath, allow_pickle=True)
    times = data["times"]
    fluxes = data["fluxes"]
    flux_errors = data["flux_errs"]  # unused so far
    filters = data["filters"]
    classes = None
    if plot_classes:
        classes = data["ids"]

    # move this to the cleaning script
    for i, e_times in enumerate(times):
        e_times = [t - e_times[0] for t in e_times]
        times[i] = np.array(e_times)

    # TODO: add zero-flux points for each passband for times outside of expressed range
    # (since light actually goes 0 --> values --> 0)
    # this will change the results greatly and require polynomial fitting; the current script is just a proof of concept

    if to_analyze == -1:
        to_analyze = len(times)
    print("Analyzing {} light curves".format(to_analyze))

    rows = []
    for i in range(to_analyze):
        features = get_features(i, times, fluxes, filters)
        rows.append(features)
        if plot_all:
            plot_event(features, i, times, fluxes, filters, plot_regression=False)

    X = np.array(rows)

    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    # clustering goes here
    model.fit(X_scaled)
    labels = dbscan.labels_

    outliers = np.where(labels == -1)[0]  # np.where() returns a length-1 tuple where only the first value is the actual output??? For some reason???
    if plot_outliers:
        for o in outliers:
            plot_event(X[o, :], o, times, fluxes, filters)

    # plot clusters in 2D space
    clusters, cl_ixes, cl_counts = np.unique(labels, return_index=True, return_counts=True)
    print("CLUSTERS:")
    print(clusters)
    print(cl_counts)

    f1 = 3  # teal intercept
    f2 = 11  # red intercept
    feature1 = X[:, f1]
    feature2 = X[:, f2]

    # plot clusters
    fig, ax = plt.subplots()
    for label in clusters:
        ixes = np.where(labels == label)[0]
        points1 = feature1[ixes]
        points2 = feature2[ixes]
        ax.scatter(points1, points2, label=label)

    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.xlabel("Red Passband IR Intercept")
    plt.ylabel("Teal Passband IR Intercept")
    ax.legend()
    plt.show()

    # plot labels
    if plot_classes:
        class_ids, class_ixes, class_counts = np.unique(classes[:to_analyze], return_index=True, return_counts=True)

        print("CLASSES:")
        print(class_ids)
        print(class_counts)

        fig, ax = plt.subplots()
        for cid in class_ids:
            ixes = np.where(classes[:to_analyze] == cid)[0]
            points1 = feature1[ixes]
            points2 = feature2[ixes]
            ax.scatter(points1, points2, label=CLASS_NAMES[int(cid)])

        plt.xscale("symlog")
        plt.yscale("symlog")
        plt.xlabel("Red Passband IR Intercept")
        plt.ylabel("Teal Passband IR Intercept")
        ax.legend()
        plt.show()

    matrix = metrics.cluster.contingency_matrix(classes[:to_analyze], labels)
    df = pd.DataFrame(matrix, index=class_ids, columns=clusters)
    print(df)
    print("Normalized mutual info score:")
    print(metrics.normalized_mutual_info_score(classes[:to_analyze], labels))


if __name__ == '__main__':
    dbscan = DBSCAN(eps=0.7, min_samples=24)

    main(dbscan, FULL_FILEPATH, to_analyze=50000, plot_all=False, plot_outliers=False, plot_classes=True)
