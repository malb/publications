# -*- coding: utf-8 -*-

from sage.all import shuffle, randint, ceil, next_prime, log, cputime, mean, variance, set_random_seed, sqrt
from copy import copy
from sage.all import GF, ZZ
from sage.all import random_matrix, random_vector, vector, matrix, identity_matrix
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler \
    as DiscreteGaussian
from estimator.estimator import preprocess_params, stddevf


def gen_fhe_instance(n, q, alpha=None, h=None, m=None, seed=None):
    """
    Generate FHE-style LWE instance

    :param n:     dimension
    :param q:     modulus
    :param alpha: noise rate (default: 8/q)
    :param h:     hamming weight of the secret (default: 2/3n)
    :param m:     number of samples (default: n)

    """
    if seed is not None:
        set_random_seed(seed)

    q = next_prime(ceil(q)-1, proof=False)
    if alpha is None:
        alpha = ZZ(8)/q

    n, alpha, q = preprocess_params(n, alpha, q)

    stddev = stddevf(alpha*q)

    if m is None:
        m = n
    K = GF(q, proof=False)
    A = random_matrix(K, m, n)

    if h is None:
        s = random_vector(ZZ, n, x=-1, y=1)
    else:
        S = [-1, 1]
        s = [S[randint(0, 1)] for i in range(h)]
        s += [0 for _ in range(n-h)]
        shuffle(s)
        s = vector(ZZ, s)
    c = A*s

    D = DiscreteGaussian(stddev)

    for i in range(m):
        c[i] += D()

    return A, c


def dual_instance0(A):
    """
    Generate dual attack basis.

    :param A: LWE matrix A

    """
    q = A.base_ring().order()
    B0 = A.left_kernel().basis_matrix().change_ring(ZZ)
    m = B0.ncols()
    n = B0.nrows()
    r = m-n
    B1 = matrix(ZZ, r, n).augment(q*identity_matrix(ZZ, r))
    B = B0.stack(B1)
    return B


def dual_instance1(A, scale=1):
    """
    Generate dual attack basis for LWE normal form.

    :param A: LWE matrix A

    """
    q = A.base_ring().order()
    n = A.ncols()
    B = A.matrix_from_rows(range(0, n)).inverse().change_ring(ZZ)
    L = identity_matrix(ZZ, n).augment(B)
    L = L.stack(matrix(ZZ, n, n).augment(q*identity_matrix(ZZ, n)))

    for i in range(0, 2*n):
        for j in range(n, 2*n):
            L[i, j] = scale*L[i, j]

    return L


def balanced_lift(e):
    """
    Lift e mod q to integer such that result is between -q/2 and q/2

    :param e: a value or vector mod q

    """
    from sage.rings.finite_rings.integer_mod import is_IntegerMod

    q = e.base_ring().order()
    if is_IntegerMod(e):
        e = ZZ(e)
        if e > q//2:
            e -= q
        return e
    else:
        return vector(balanced_lift(ee) for ee in e)


def apply_short1(y, A, c, scale=1):
    """
    Compute `y*A`, `y*c` where y is a vector in the integer row span of
    ``dual_instance(A)``

    :param y: (short) vector in scaled dual lattice
    :param A: LWE matrix
    :param c: LWE vector
    """
    m = A.nrows()
    y = vector(ZZ, 1/ZZ(scale) * y[-m:])
    a = balanced_lift(y*A)
    e = balanced_lift(y*c)
    return a, e


def log_mean(X):
    return log(mean([abs(x) for x in X]), 2)


def log_var(X):
    return log(variance(X).sqrt(), 2)


def silke(A, c, beta, h, m=None, scale=1, float_type="double"):
    """

    :param A:    LWE matrix
    :param c:    LWE vector
    :param beta: BKW block size
    :param m:    number of samples to consider
    :param scale: scale rhs of lattice by this factor

    """
    from fpylll import BKZ, IntegerMatrix, LLL, GSO
    from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

    if m is None:
        m = A.nrows()

    L = dual_instance1(A, scale=scale)
    L = IntegerMatrix.from_matrix(L)
    L = LLL.reduction(L, flags=LLL.VERBOSE)
    M = GSO.Mat(L, float_type=float_type)
    bkz = BKZ2(M)
    t = 0.0
    param = BKZ.Param(block_size=beta,
                      strategies=BKZ.DEFAULT_STRATEGY,
                      auto_abort=True,
                      max_loops=16,
                      flags=BKZ.VERBOSE|BKZ.AUTO_ABORT|BKZ.MAX_LOOPS)
    bkz(param)
    t += bkz.stats.total_time

    H = copy(L)

    import pickle
    pickle.dump(L, open("L-%d-%d.sobj"%(L.nrows, beta), "wb"))

    E = []
    Y = set()
    V = set()
    y_i = vector(ZZ, tuple(L[0]))
    Y.add(tuple(y_i))
    E.append(apply_short1(y_i, A, c, scale=scale)[1])

    v = L[0].norm()
    v_ = v/sqrt(L.ncols)
    v_r = 3.2*sqrt(L.ncols - A.ncols())*v_/scale
    v_l = sqrt(h)*v_

    fmt = u"{\"t\": %5.1fs, \"log(sigma)\": %5.1f, \"log(|y|)\": %5.1f, \"log(E[sigma]):\" %5.1f}"

    print
    print fmt%(t,
               log(abs(E[-1]), 2),
               log(L[0].norm(), 2),
               log(sqrt(v_r**2 + v_l**2), 2))
    print
    for i in range(m):
        t = cputime()
        M = GSO.Mat(L, float_type=float_type)
        bkz = BKZ2(M)
        t = cputime()
        bkz.randomize_block(0, L.nrows, stats=None, density=3)
        LLL.reduction(L)
        y_i = vector(ZZ, tuple(L[0]))
        l_n = L[0].norm()
        if L[0].norm() > H[0].norm():
            L = copy(H)
        t = cputime(t)

        Y.add(tuple(y_i))
        V.add(y_i.norm())
        E.append(apply_short1(y_i, A, c, scale=scale)[1])
        if len(V) >= 2:
            fmt =  u"{\"i\": %4d, \"t\": %5.1fs, \"log(|e_i|)\": %5.1f, \"log(|y_i|)\": %5.1f,"
            fmt += u"\"log(sigma)\": (%5.1f,%5.1f), \"log(|y|)\": (%5.1f,%5.1f), |Y|: %5d}"
            print fmt%(i+2, t, log(abs(E[-1]), 2), log(l_n, 2), log_mean(E), log_var(E), log_mean(V), log_var(V), len(Y))

    return E
