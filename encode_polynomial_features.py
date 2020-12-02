import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


NUM_PASSBANDS = 6


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


def main(filepath, to_analyze):

    # read in data
    data = np.load(filepath, allow_pickle=True)
    times = data["times"]
    fluxes = data["fluxes"]
    flux_errors = data["flux_errs"]  # unused so far
    filters = data["filters"]
    classes = data["ids"]

    # todo: add 0-flux points for each passband

    if to_analyze == -1:
        to_analyze = len(times)
    print("Analyzing {} light curves from {}".format(to_analyze, filepath))

    rows = []
    for i in range(to_analyze):
        features = get_features(i, times, fluxes, filters)
        rows.append(features)

    X = np.array(rows)
    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    X = np.append(X, classes[:to_analyze].reshape(-1, 1), 1)
    X_scaled = np.append(X_scaled, classes[:to_analyze].reshape(-1, 1), 1)

    # output new feature array to file
    np.savetxt("polynomial_features.csv", X, delimiter=",")
    np.savetxt("polynomial_features_scaled.csv", X_scaled, delimiter=",")


if __name__ == "__main__":
    print("Encoding data into features...")

    # give the filepath to the data file in the first argument and the row count in the second
    main(sys.argv[1], int(sys.argv[2]))
