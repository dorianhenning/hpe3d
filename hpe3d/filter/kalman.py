import numpy as np
from filterpy.kalman import KalmanFilter as KF

FACTOR = {'c': 1,
          'v': 2,
          'a': 3}


def filter_variable(x, mode, f1=1., f2=1.):
    '''
        filters position measurement vector x with a
        Kalman filter with a "uncertainty" of d
        input:
            x:      B x 3 position vector
        return:
            x_filt: B x 3 filtered position vector

    '''
    assert x.ndim == 2
    batch_size, ndim = x.shape
    x_filt = np.empty_like(x)

    x_cov = np.cov(x, rowvar=False) * np.eye(ndim)  # eliminate off-diagonal elements of R
    kf = KalmanFilter(x_cov, x[0, :], mode=mode, f1=f1, f2=f2)

    for i in range(batch_size):
        _ = kf.predict()
        kf.update(x[i])

        x_filt[i] = kf.get_state()[:ndim, 0]

    return x_filt


class KalmanFilter():

    def __init__(
            self,
            R,
            z_init,
            mode,
            f1,
            f2,
            dt=1.
    ):
        assert mode in ['c', 'v', 'a']
        self.factor = FACTOR[mode]
        self.dim_z = len(z_init)
        self.dim_x = self.dim_z * self.factor

        self._kf = KF(dim_x=self.dim_x, dim_z=self.dim_z)

        self._kf.F = np.eye(self.dim_x, dtype=float)
        self._kf.H = np.eye(self.dim_x, dtype=float)[:self.dim_z]
        self._kf.R = R
        self._kf.P *= 10. * f1
        self._kf.Q *= .01 * f2

        if self.factor > 1:
            n1 = (self.factor - 1) * self.dim_z
            shift_1 = self.dim_z
            self._kf.F += np.diag(np.tile(dt, n1), k=shift_1)
            self._kf.P[shift_1:, shift_1:] *= 1000.
            self._kf.Q[shift_1:, shift_1:] *= .01

            if self.factor > 2:
                d2 = dt * dt
                n2 = (self.factor - 2) * self.dim_z
                shift_2 = 2 * shift_1
                self._kf.F += np.diag(np.tile(d2, n2), k=shift_2)
                self._kf.P[shift_1:, shift_1:] *= 10.
                self._kf.Q[shift_1:, shift_1:] *= .1

        self._kf.x[:self.dim_z, 0] = z_init
        self.history = []

    def update(self, z):
        self._kf.update(z)

    def predict(self):
        self._kf.predict()
        self.history.append(self._kf.x)

        return self.history[-1]

    def get_state(self):

        return self._kf.x
