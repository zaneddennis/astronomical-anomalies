import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN


DATA_FILEPATH = "notebooks/mockdata.npz"
NUM_PASSBANDS = 6


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
            linreg.fit(X[:, 0:1], X[:, 1:2])

            features[passband * 2] = linreg.coef_[0][0]
            features[1 + passband * 2] = linreg.intercept_[0]

    return features


def main(model, to_analyze=-1, plot_all=False, plot_outliers=False):
    data = np.load(DATA_FILEPATH, allow_pickle=True)
    times = data["times"]
    fluxes = data["fluxes"]
    flux_errors = data["flux_errs"]  # unused so far
    filters = data["filters"]

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
    #print(X)

    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    # clustering goes here
    model.fit(X_scaled)
    labels = dbscan.labels_

    outliers = np.where(labels == -1)[0]  # np.where() returns a length-1 tuple where only the first value is the actual output??? For some reason???
    if plot_outliers:
        for o in outliers:
            plot_event(X[o, :], o, times, fluxes, filters)

    clusters, cl_counts = np.unique(labels, return_counts=True)
    print("CLUSTERS:")
    print(clusters)
    print(cl_counts)

    # plot clusters in 2D space
    f1 = 2  # teal slope
    f2 = 10  # red slope
    feature1 = X[:, f1]
    feature2 = X[:, f2]



    fig, ax = plt.subplots()
    for label in clusters:
        ixes = np.where(labels == label)[0]
        points1 = feature1[ixes]
        points2 = feature2[ixes]
        ax.scatter(points1, points2, label=label)

    plt.xlabel("Red Passband IR Slope")
    plt.ylabel("Teal Passband IR Slope")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    dbscan = DBSCAN(eps=1.5, min_samples=5)

    main(dbscan, plot_all=False, plot_outliers=False)
