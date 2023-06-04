import numpy as np
import scipy as sci
from scipy import linalg
import scipy.sparse.linalg as scislin

from .errors.sdmd_errors import ATypeError, SVDAlgorythmError, ModsTypeError
from .constants.sdmd import ModsTypes, SVDTypes
from .utils.sdmd import rT, cT


def sdmd(A, dt=1, k: int = None, modes: ModsTypes = ModsTypes.EXACT.value,
         return_amplitudes: bool = False, return_vandermonde: bool = False,
         svd: SVDTypes = SVDTypes.TRUNCATED.value, order: bool = True):
    m, n = A.shape
    dat_type = A.dtype

    if dat_type == np.float32:
        fT = rT
    elif dat_type == np.float64:
        fT = rT
    elif dat_type == np.complex64:
        fT = cT
    elif dat_type == np.complex128:
        fT = cT
    else:
        raise ATypeError('A.dtype is not supported')

    X = A[:, range(0, n - 1)]
    Y = A[:, range(1, n)]

    if k is not None:
        if svd == SVDTypes.PARTIAL.value:
            U, s, Vh = scislin.svds(X, k=k)
            U[:, :k] = U[:, k - 1::-1]
            s = s[::-1]
            Vh[:k, :] = Vh[k - 1::-1, :]

        elif svd == SVDTypes.TRUNCATED.value:
            U, s, Vh = sci.linalg.svd(X, compute_uv=True,
                                      full_matrices=False,
                                      overwrite_a=False,
                                      check_finite=True)
            U = U[:, range(k)]
            s = s[range(k)]
            Vh = Vh[range(k), :]

        else:
            raise SVDAlgorythmError('SVD algorithm is not supported')
    else:
        U, s, Vh = sci.linalg.svd(X, compute_uv=True,
                                  full_matrices=False,
                                  overwrite_a=False,
                                  check_finite=True)

    Vscaled = fT(Vh) * s ** -1
    G = np.dot(Y, Vscaled)
    M = np.dot(fT(U), G)

    l, W = sci.linalg.eig(M, right=True, overwrite_a=True)

    omega = np.log(l) / dt

    if order:
        sort_idx = np.argsort(np.abs(omega))
        W = W[:, sort_idx]
        l = l[sort_idx]
        omega = omega[sort_idx]

    if modes == ModsTypes.STANDARD.value:
        F = np.dot(U, W)
    elif modes == ModsTypes.EXACT.value:
        F = np.dot(G, W)
    elif modes == ModsTypes.EXACT_SCALED.value:
        F = np.dot((1 / l) * G, W)
    else:
        raise ModsTypeError('Type of modes is not supported, choose STANDARD, EXACT or EXACT_SCALED')

    if return_amplitudes and return_vandermonde:
        b, _, _, _ = sci.linalg.lstsq(F, A[:, 0])
        V = np.fliplr(np.vander(l, N=n))
        return F, b, V, omega
    elif return_amplitudes and not return_vandermonde:
        b, _, _, _ = sci.linalg.lstsq(F, A[:, 0])
        return F, b, omega
    else:
        return F, omega
