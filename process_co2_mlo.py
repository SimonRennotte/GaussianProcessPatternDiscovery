import math

import Gp_pattern
import Gp_se
import numpy as np
from utils import read_co2_mlo, generate_sync_pattern, split_train_val_test, generate_array_pred, plot_results


def main():
    x, y = read_co2_mlo()
    #x, y = generate_sync_pattern(x_min=-15, x_max=15, pattern_dist=10, x_step=0.25)
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y, prop_train=0.8, prop_val=0.0)

    gp = Gp_pattern.GpPatternDiscovery(x_train, y_train, n_gm=10, sn_range_init=(1e-3, 1), sn_range_limits=(1e-3, math.inf), max_f_init=0.2)

    """gp = Gp_se.GpSe(x_train, y_train, l_range_init=((0., 1.),), sf_range_init=(0, 1.), sn_range_init=(1e-3, 1),
                                    l_range_limits=((0., math.inf),), sf_range_limits=(0., math.inf), sn_range_limits=(1e-3, math.inf))"""

    # gp.train_hyperparams(x_val=None, y_val=None, n_iters=250, n_restarts=5, lr=3e-3, prop_in=0.5)
    gp.train_hyperparams(x_val=None, y_val=None, n_iters=250, n_restarts=5, lr=1e-3, prop_in=0.4)

    x_pred = generate_array_pred(x_max=np.max((x_train.max(), x_test.max())), x_min=np.min((x_train.min(), x_test.min())), border_prop=0.1)

    y_pred, y_pred_cov = gp.predict(xs=x_pred)

    plot_results(x_train, x_val, x_test, x_pred, y_train, y_val, y_test, y_pred, y_pred_cov)

    gp.plot_spectral_density(max_f=1 / (x_train[1:] - x_train[:-1]).min(0))

    gp.plot_cov_fct(x=x_train)


if __name__ == "__main__":
    main()
