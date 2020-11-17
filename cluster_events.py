import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN


DATA_FILEPATH = "notebooks/mockdata.npz"
NUM_PASSBANDS = 6


def plot_event(event_id, times, fluxes, filters):
    time_row = times[event_id]
    flux_row = fluxes[event_id]
    filter_row = filters[event_id]

    for passband, color in zip(range(NUM_PASSBANDS), ("blue", "cyan", "green", "yellow", "orange", "red")):
        good_ixes = np.where(filter_row == str(passband))
        plt.plot(time_row[good_ixes], flux_row[good_ixes], color=color, marker="o")

    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.ylim(-8.0, 0.1)
    plt.title('Light Curve {}'.format(event_id))
    plt.show()


def plot_fitted_event(params, event_id, times, fluxes, filters):
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
        for i, t in enumerate(times[event_id]):
            f = slope * t + intercept
            points[i] = [t, f]

        plt.plot(points[:, 0], points[:, 1], color=color, linestyle="dashed")

    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.ylim(-8.0, 0.1)
    plt.title('Intermediate Regression Curve {}'.format(event_id))
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


def main(model, to_analyze=10, plot_all=False, plot_outliers=False):
    data = np.load(DATA_FILEPATH, allow_pickle=True)
    times = data["times"]
    fluxes = data["fluxes"]
    flux_errors = data["flux_errs"]  # unused so far
    filters = data["filters"]

    for i, e_times in enumerate(times):
        e_times = [t - e_times[0] for t in e_times]
        times[i] = np.array(e_times)

    # TODO: add zero-flux points for each passband for all other times

    rows = []
    for i in range(to_analyze):
        features = get_features(i, times, fluxes, filters)
        rows.append(features)
        if plot_all:
            plot_event(i, times, fluxes, filters)
            plot_fitted_event(features, i, times, fluxes, filters)

    X = np.array(rows)
    #print(X)

    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    # clustering goes here
    model.fit(X_scaled)
    labels = dbscan.labels_

    print("OUTLIERS:")
    for i, l in enumerate(labels):
        if l == -1:
            print(i)
            if plot_outliers:
                plot_fitted_event(X[i, :], i, times, fluxes, filters)


if __name__ == '__main__':
    dbscan = DBSCAN(eps=3.0)

    main(dbscan, to_analyze=100, plot_outliers=True)
