import numpy as np

_metric_spaces = {}
def get_metric_space(name, dim):
    return _metric_spaces[name](dim)

def _register_name(name):
    def _registerer(cls):
        _metric_spaces[name] = cls
        return cls
    return _registerer


def _normalize_vectors(vec, axis=-1, ord=2, eps=1e-8, norm=None):
    vec = np.asarray(vec, float)
    if norm is None:
        norm = np.linalg.norm(vec, axis=axis, ord=ord, keepdims=True)
    norm = np.broadcast_to(norm, vec.shape)

    ret = np.zeros_like(vec)
    cond = norm>eps
    ret[cond] = vec[cond] / norm[cond]

    return ret


class QuasimetricSpaceBase(object):
    """ Base class for quasimetric space
    """
    extra_dim = 0


    def __init__(self, dim):
        self.dim = dim
        self.nomial_dim = dim + self.extra_dim

    def compute_distance(self, u, v, axis=-1):
        raise NotImplementedError()

    def compute_distance_grad_u(self, u, v, axis=-1):
        raise NotImplementedError()

    def compute_distance_grad_v(self, u, v, axis=-1):
        """ gradient of distance function for second argument
        In the case of symmetric distance, it is just a exchange of arguments.
        """
        return self.compute_distance_grad_u(v, u, axis=axis)

    def exp_map(self, x, v, axis=-1):
        return x + v

    def clip_vectors(self, vectors, axis=-1):
        """Clip vectors to be in the feasible space.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D or more.
        axis : int
            An axis which represents vector dimensionality

        Returns
        -------
        numpy.array
            Array clipped in the feasible manifold
        """
        return vectors

    def clip_gradients(self, vectors, grad_vectors, axis=-1):
        """Clip gradient vectors to be in a tangent space of the manifold.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D or more.
        grad_vectors : numpy.array
            Can be 1-D or more.

        Returns
        -------
        numpy.array
            Array clipped in the tangent space
        """
        return grad_vectors


@_register_name("euc")
class EuclideanSpace(QuasimetricSpaceBase):
    def compute_distance(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return np.linalg.norm(x-y, axis=axis)
        
    def compute_distance_grad_u(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return _normalize_vectors(x-y, axis=axis)

@_register_name("l1")
class L1NormSpace(QuasimetricSpaceBase):
    def compute_distance(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return np.linalg.norm(x-y, axis=axis, ord=1)
        
    def compute_distance_grad_u(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        return np.sign(x-y)

@_register_name("unif")
class UniformNormSpace(QuasimetricSpaceBase):
    def compute_distance(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return np.linalg.norm(x-y, axis=axis, ord=np.inf)
        
    def compute_distance_grad_u(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        axy = np.abs(x-y)
        ismax = (axy == axy.max(axis=axis, keepdims=True))
        return np.sign(x-y)*ismax / np.sum(ismax, axis=axis, keepdims=True).astype(float)


@_register_name("sphere")
class SphericalSpace(QuasimetricSpaceBase):
    extra_dim = 1
    def compute_distance(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        norm_x = np.linalg.norm(x, axis=axis)
        norm_y = np.linalg.norm(y, axis=axis)
        norm_xy = np.linalg.norm(x-y, axis=axis)

        cos_theta = (norm_x**2 + norm_y**2 - norm_xy**2) / (2 * norm_x * norm_y)
        return np.arccos(np.clip(cos_theta, -1,1))

    def compute_distance_grad_u(self, x, y, axis=-1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        dot_xy = np.sum(x*y, axis=axis, keepdims=True)
        norm_x = np.sum(x*x, axis=axis, keepdims=True)

        return _normalize_vectors((dot_xy / norm_x**2) * x - y, axis=axis)

    def clip_vectors(self, x, axis=-1):
        return _normalize_vectors(x, axis=axis)

    def clip_gradients(self, x, v, axis=-1):
        v = np.asarray(v, float)
        x = _normalize_vectors(x, axis=axis)
        vx = np.sum(v*x, axis=axis, keepdims=True)

        return v - x * vx

    def exp_map(self, x, v, axis=-1):
        x = self.clip_vectors(x, axis=axis)
        v = self.clip_gradients(x, v, axis=axis)
        v_norm = _normalize_vectors(v, axis=axis)
        dist = np.linalg.norm(v, axis=axis, keepdims=True)

        return x*np.cos(dist) + v_norm*np.sin(dist)



@_register_name("simplex")
class SimplexNormSpace(QuasimetricSpaceBase):
    def compute_distance(self, x, y, axis=-1):
        x = np.asarray(x)
        y = np.asarray(y)

        delta = np.swapaxes(x-y, axis, -1).dot(self._simplex_vertices).swapaxes(axis, -1)
        return delta.max(axis=axis)

    def compute_distance_grad_u(self, x, y, axis=-1):
        x = np.asarray(x)
        y = np.asarray(y)

        delta = np.swapaxes(x-y, axis, -1).dot(self._simplex_vertices).swapaxes(axis, -1)

        ismax = (delta == delta.max(axis=axis, keepdims=True))


        return delta.max(axis=axis)

    @property
    def _simplex_vertices(self):
        if not hasattr(self, "__simplex_vertices"):
            n = self.dim+1
            P = np.eye(n) - np.ones((n, n)) / n

            U, _ = np.linalg.qr(P[:,:-1])
            subspace_basis = U.T

            # e, U = np.linalg.eigh(P)
            # subspace_basis = U[:,1:].T

            self.__simplex_vertices = np.dot(subspace_basis, P)

        return self.__simplex_vertices


@_register_name("hyp")
class HyperbolicSpace(QuasimetricSpaceBase):
    extra_dim = 1

    @staticmethod
    def _lorentzian_prod(x, y, axis=-1, keepdims=False):
        xy_0, xy_other = np.split(np.multiply(x,y), [1], axis=axis)
        return np.sum(xy_other, axis=axis, keepdims=keepdims) - np.sum(xy_0, axis=axis, keepdims=keepdims)

    def compute_distance(self, x, y, axis=-1, keepdims=False):
        return np.arccosh(-self._lorentzian_prod(x, y, axis=axis, keepdims=keepdims))

    def clip_vectors(self, x, axis=-1):
        x_0, x_other = np.split(x, [1], axis=axis)
        x_0 = np.sqrt(1 + np.sum(x_other**2, axis=axis, keepdims=True))
        return np.concatenate((x_0, x_other), axis=axis)

    def clip_gradients(self, x, v, axis=-1):
        xx_l = self._lorentzian_prod(x, x, axis=axis, keepdims=True)
        xv_l = self._lorentzian_prod(x, v, axis=axis, keepdims=True)
        return v - x * (xv_l/xx_l)


    def compute_distance_grad_u(self, x, y, axis=-1):
        EPS = 1e-8

        # calculate gradient by x
        dist = self.compute_distance(x, y, axis=axis, keepdims=True)
        denom = np.sinh(dist)
        h = _normalize_vectors(-y, axis=axis, norm=denom)

        # project into tangent space
        return self.clip_gradients(x, h, axis=axis)

    def exp_map(self, x, v, axis=-1):
        x = self.clip_vectors(x, axis=axis)
        v = self.clip_gradients(x, v, axis=axis)

        v_l = np.sqrt(self._lorentzian_prod(v, v, keepdims=True))
        v_norm = _normalize_vectors(v, norm=v_l, axis=axis)

        return np.cosh(v_l)*x + np.sinh(v_l)*v_norm



