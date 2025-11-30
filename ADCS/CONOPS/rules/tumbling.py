import numpy as np
from ADCS.CONOPS.rulebook import Rule

def is_tumbling(t, x_hat, est_sat, est_os):
    omega = x_hat[0:3]
    w = np.linalg.norm(omega)

    TUMBLE_THRESHOLD = np.deg2rad(0.5) #rad/s 

    return w > TUMBLE_THRESHOLD