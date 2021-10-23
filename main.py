import math

import Gp_pattern
import Gp_se
import numpy as np
from utils import read_co2_mlo, generate_sync_pattern, split_train_val_test, generate_array_pred, plot_results


def main():
    x, y = read_co2_mlo()
    #x, y = generate_sync_pattern()

    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y)

    gp = Gp_pattern.GpPatternDiscovery(x_train, y_train)
    # gp = Gp_se.GpSe(x_train, y_train)

    gp.train(x_val=x_val, y_val=y_val)

    x_pred = generate_array_pred(x_max=x_test.max(), x_min=x_train.min(), border_prop=0.1)

    y_pred, y_pred_cov = gp.predict(xs=x_pred)

    plot_results(x_train, x_val, x_test, x_pred, y_train, y_val, y_test, y_pred, y_pred_cov)

    gp.plot_spectral_density(max_f=1 / (x_train[1:] - x_train[:-1]).min(0))

    gp.plot_cov_fct(x=x_train)


if __name__ == "__main__":
    main()
