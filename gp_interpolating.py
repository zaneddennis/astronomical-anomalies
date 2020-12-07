import sys
from random import *

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import george


NUM_PASSBANDS = 6
OUTPUT_POINTS = 64


def plot_event(e_times, e_fluxes, e_filters, event_id, pred, x_times):
    for passband, color in zip(range(NUM_PASSBANDS), ("blue", "cyan", "green", "yellow", "orange", "red")):
        good_ixes = np.where(e_filters == passband)
        plt.plot(e_times[good_ixes], e_fluxes[good_ixes], color=color, marker="o")

    for passband, color in zip(range(NUM_PASSBANDS), ("blue", "cyan", "green", "yellow", "orange", "red")):
        plt.plot(x_times, pred[:,passband], color=color, marker="x", ls="--")

    plt.ylim(-0.1, np.amax(pred)+0.5)
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.title("Light Curve {}".format(event_id))
    plt.show()


def make_dense_lc(e_times, e_fluxes, e_filters):
    stacked_data = np.vstack([e_times, e_filters]).T
    #x_pred = np.zeros((len(e_times) * NUM_PASSBANDS, 2))
    x_pred = np.zeros((OUTPUT_POINTS*6, 2))
    kernel = np.var(e_fluxes) * george.kernels.ExpSquaredKernel([100, 1], ndim=2)
    gp = george.GP(kernel)
    gp.compute(stacked_data, 0)

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(e_fluxes)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(e_fluxes)

    result = optimize.minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)

    new_times = np.linspace(0, 100, OUTPUT_POINTS)
    for jj, time in enumerate(new_times):
        x_pred[jj * NUM_PASSBANDS:jj * NUM_PASSBANDS + NUM_PASSBANDS, 0] = [time] * NUM_PASSBANDS
        x_pred[jj * NUM_PASSBANDS:jj * NUM_PASSBANDS + NUM_PASSBANDS, 1] = np.arange(NUM_PASSBANDS)
    pred, pred_var = gp.predict(e_fluxes, x_pred, return_var=True)

    return pred, pred_var, new_times


if __name__ == '__main__':
    filepath = sys.argv[1]
    to_analyze = int(sys.argv[2])

    # read data
    data = np.load(filepath, allow_pickle=True)
    times = data["times"][:to_analyze]
    fluxes = data["fluxes"][:to_analyze]
    flux_errors = data["flux_errs"][:to_analyze]  # unused so far
    filters = data["filters"][:to_analyze]
    classes = data["ids"][:to_analyze]

    output_rows = []
    output_labels = []
    for i in range(to_analyze):
        e_times = times[i]
        e_fluxes = fluxes[i]
        e_flux_errors = flux_errors[i]
        e_filters = filters[i]
        label = classes[i]

        try:
            pred, pred_vars, x_times = make_dense_lc(e_times, e_fluxes, e_filters)
            #y_fluxes = np.reshape(pred, (20, 6))
            y_fluxes = pred.reshape(OUTPUT_POINTS, 6)

            output_labels.append(label)
            output_rows.append(pred)

            if random() < 10.0/to_analyze:
                plot_event(e_times, e_fluxes, e_filters, i, y_fluxes, x_times)

        except:
            pass

    # compile into output array
    output = np.array(output_rows)
    output_labels = np.array(output_labels)
    print(output.shape)
    print(output_labels.shape)

    np.savetxt("interpolated_10k.csv", output, delimiter=",")
    np.savetxt("interpolated_10k_labels.csv", output_labels, delimiter=",")
