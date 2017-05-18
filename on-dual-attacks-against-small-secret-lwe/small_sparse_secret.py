"""
Security estimates for binary secret LWE instances.

"""
from collections import OrderedDict
from functools import partial
from copy import deepcopy
from sage.all import sqrt, log, get_verbose, binomial, ceil, RealField
from estimator.estimator import preprocess_params, stddevf, lattice_reduction_opt_m
from estimator.estimator import distinguish_required_m, cost_reorder, cost_str
from estimator.estimator import bkz_runtime_delta
from estimator.estimator import fhe_params
from estimator.estimator import sis, decode, kannan
from estimator.estimator import decode_small_secret_mod_switch_and_guess as decode_sml
from estimator.estimator import kannan_small_secret_mod_switch_and_guess as kannan_sml
from estimator.estimator import bai_gal_small_secret as bai_gal_sml
from estimator.estimator import alphaf, uniform_variance_from_bounds
from estimator.estimator import cost_repeat
from estimator.estimator import drop_and_solve, sis_drop_and_solve, decode_drop_and_solve, bai_gal_drop_and_solve
from estimator.estimator import sis_small_secret_mod_switch
from estimator.estimator import sis_applebaum, decode_applebaum

n, alpha, q = fhe_params(n=2**11, L=2)

def make_table(n, alpha, q):
    RR = alpha.parent()
    template = OrderedDict([
        ("SIS", OrderedDict([("sieve", 0), ("lp", 0)])),
        ("DEC", OrderedDict([("sieve", 0), ("lp", 0)])),
        ("Kannan", OrderedDict([("sieve", 0), ("lp", 0)]))
    ])

    strategies = OrderedDict()
    strategies["base line"]  = deepcopy(template)
    strategies["secret /"]  = deepcopy(template)
    strategies["modulus /"] = deepcopy(template)
    strategies["drop"]       = deepcopy(template)
    strategies["++"]         = deepcopy(template)

    for target in ("lp", "sieve"):
        strategies["base line"]["SIS"][target] = RR(sis(n, alpha, q, optimisation_target=target)[target])
        strategies["base line"]["DEC"][target] = RR(decode(n, alpha, q, optimisation_target=target)["rop"])
        strategies["base line"]["Kannan"][target] = RR(kannan(n, alpha, q, optimisation_target=target)[target])

    kwds = {"secret_bounds": (-1, 1), "h": 64}

    t = strategies["modulus /"]
    for target in ("lp", "sieve"):
        r = RR(sis_small_secret_mod_switch(n, alpha, q, optimisation_target=target, **kwds)[target])
        t["SIS"][target] = r

        r = RR(decode_sml(n, alpha, q, optimisation_target=target, **kwds)["rop"])
        t["DEC"][target] = r

        r = RR(kannan_sml(n, alpha, q, optimisation_target=target, **kwds)[target])
        t["Kannan"][target] = r

    t = strategies["secret /"]
    for target in ("lp", "sieve"):
        r = RR(sis_applebaum(n, alpha, q, m=2*n, optimisation_target=target, **kwds)[target])
        t["SIS"][target] = r

        r = RR(decode_applebaum(n, alpha, q, m=2*n, optimisation_target=target, **kwds)["rop"])
        t["DEC"][target] = r

        # we think of Bai-Gal as a better version of Applebaum
        r = RR(bai_gal_sml(n, alpha, q, optimisation_target=target, **kwds)[target])
        t["Kannan"][target] = r

    t = strategies["drop"]
    for target in ("lp", "sieve"):
        r = RR(sis_drop_and_solve(n, alpha, q, optimisation_target=target, postprocess=True, **kwds)[target])
        t["SIS"][target] = r

        r = RR(decode_drop_and_solve(n, alpha, q, optimisation_target=target, **kwds)["rop"])
        t["DEC"][target] = r

        r = RR(bai_gal_drop_and_solve(n, alpha, q, optimisation_target=target, **kwds)[target])
        t["Kannan"][target] = r

    t = strategies["++"]
    for target in ("lp", "sieve"):
        r = RR(drop_and_solve(sis_small_secret_mod_switch, n, alpha, q,
                              optimisation_target=target, postprocess=True, **kwds)[target])
        t["SIS"][target] = r

        r = RR(drop_and_solve(decode_applebaum, n, alpha, q, m=2*n, optimisation_target=target, **kwds)["rop"])
        t["DEC"][target] = r

        r = RR(drop_and_solve(bai_gal_sml, n, alpha, q, optimisation_target=target, **kwds)[target])
        t["Kannan"][target] = r

    table = [["", "Strategy", "Dual",   "",   "Dec",    "",   "Embed", ""],
             ["",  "",         "sieve", "lp", "sieve", "lp", "sieve", "lp"]]

    for i, (name, row) in enumerate(strategies.items()):
        r = ["%d"%i, name]
        for algorithm in row.values():
            for target in algorithm.values():
                r.append("%5.1f"%log(target, 2.0))
        table.append(r)

    for row in table:
        print "|".join(["%10s"%entry for entry in row])

    return table, strategies


def helib_find_q(n, target_cost, helib_offset=log(2.33 * 10**9, 2), q=None, step_size=2, sigma=8.0):
    if q is None:
        q = n**2
    while True:
        cost = sis(n, sigma/q, q, optimisation_target="lp")
        if log(cost["lp"], 2.) - helib_offset < target_cost:
            return q/step_size
        else:
            q *= step_size


def make_helib_table_2(N=(1024, 2048, 4096, 8192, 16384), security=80, sigma=8.0, optimisation_target="sieve"):
    q = 1024**2

    params = []

    data = OrderedDict()

    for n in N:
        q = helib_find_q(n, security, q=q)
        print "%6.1f &"%log(q, 2.0).n(),
        params.append((n, sigma/q, q))
    print

    for param in params:
        n, alpha, q= param
        res = sis(n, alpha, q, optimisation_target=optimisation_target)
        data[(n, alpha, q, "dual")] = res
        print "%6.1f &"%log(res[optimisation_target], 2.0),
    print

    for param in params:
        n, alpha, q = param
        res = drop_and_solve(sis_small_secret_mod_switch, n, alpha, q, secret_bounds=(-1, 1), h=64,
                             optimisation_target=optimisation_target, postprocess=True)
        data[(n, alpha, q, "sparse")] = res
        print "%6.1f &"%log(res["rop"], 2.0),
    print
    return data


def make_seal_table_2(our_q=False, optimisation_target="sieve"):
    if our_q:
        params = [(1024, 47.9), (2048, 89.0), (4096, 170.8), (8192, 331.8), (16384, 652.2)]
    else:
        params = [(1024, 47.9), (2048, 94.0), (4096, 190.0), (8192, 383.0), (16384, 767.0)]

    data = OrderedDict()

    for param in params:
        n, q= param
        q = ceil(2**q)
        alpha = 8.0/q
        res = sis(n, alpha, q, optimisation_target=optimisation_target)
        data[(n, alpha, q, "dual")] = res
        print "%6.1f &"%log(res[optimisation_target], 2.0),
    print

    for param in params:
        n, q = param
        q = ceil(2**q)
        alpha = 8.0/q

        res_lll = sis_small_secret_mod_switch(n, alpha, q, secret_bounds=(-1, 1), h=ceil(2*n/3),
                                              optimisation_target=optimisation_target, use_lll=True)
        res     = sis_small_secret_mod_switch(n, alpha, q, secret_bounds=(-1, 1), h=ceil(2*n/3),
                                              optimisation_target=optimisation_target, use_lll=False)
        if res_lll[optimisation_target] < res[optimisation_target]:
            res = res_lll
        data[(n, alpha, q, "sparse")] = res
        print "%6.1f &"%log(res[optimisation_target], 2.0),
    print

    return data
