from sage.all import var, pi, assume, sqrt, log, ZZ, assume

n, alpha, q, h, m = var("n, alpha, q, h, m")
assume(n>1)
assume(n, "integer")
assume(q>2)
assume(q, "integer")
assume(alpha>0)
assume(alpha<1)
assume(h, "integer")
assume(h>0)
assume(h<=n)


def symbolic_modulus_switching(n, alpha, q, h, m=None, epsilon=None):
    if epsilon is None:
        epsilon = var("epsilon")
        assume(epsilon>0)
        assume(epsilon<ZZ(1)/2)

    delta_0 = var("delta_0")
    assume(delta_0>=1.0)

    if m is None:
        m = sqrt(n*log(q)/log(delta_0))

    e = alpha*q/sqrt(2*pi)

    c = e * sqrt(m-n)/sqrt(h)

    v = delta_0**m * (q/c)**(n/m)  # norm of the vector
    v_ = v**2/m  # variance of each component

    v_r = (m-n) * e**2 *v_   # noise contribution
    v_l = h * v_ * c**2  # nose contribution of rounding noise

    # epsilon = exp(-pi*(|v|^2/q^2))

    f = log(1/epsilon)/pi == (v_l + v_r)/q**2

    # solve
    f = 2* q**2 * m * f * pi
    f = f.simplify_full()
    f = f.solve(delta_0**(2*m))[0]
    f = f.log().canonicalize_radical()
    f = f.solve(log(delta_0))[0]
    f = f.simplify_log()
    return f


def symbolic_sis(n, alpha, q, m=None, epsilon=None):
    if epsilon is None:
        epsilon = var("epsilon")
        assume(epsilon>0)
        assume(epsilon<ZZ(1)/2)
    delta_0 = var("delta_0")
    assume(delta_0>=1.0)

    e = alpha*q/sqrt(2*pi)

    if m is None:
        m = sqrt(n*log(q)/log(delta_0))

    v = e * delta_0**m * q**(n/m)  # norm of the vector

    # epsilon = exp(-pi*(|v|^2/q^2))
    f = log(1/epsilon)/pi == (v/q)**2

    # solve
    f = 2* q**2 * m * f * pi
    f = f.simplify_full()
    f = f.solve(delta_0**(2*m))[0]
    f = f.log().canonicalize_radical()
    f = f.solve(log(delta_0))[0]
    f = f.simplify_log()
    return f


def asymptotic_simplify_term(term, x):
    """
    Recursively drop additive terms which are dominated asymptotically by other terms.

    :param term: term to process
    :param x: variable going to infinity

    :returns: a simpler term with the same asymptotic behaviour

    .. note::

        This should be replaced by Sage's asymptotic ring.
    """
    from sage.symbolic.operators import add_vararg, mul_vararg
    if term.operator() ==  add_vararg:
        ret = []
        for op1 in term.operands():
            for op2 in term.operands():
                # akward notation to make limit(foo, kappa=infinity) work
                if limit(abs(op1)/abs(op2), **{str(x):Infinity}) < 1:
                    break
            else:
                ret.append(asymptotic_simplify_term(op1, x))
        ret = add_vararg(*ret)
    elif term.operator() == mul_vararg:
        ret = mul_vararg(*[asymptotic_simplify_term(op, x) for op in term.operands()])
    else:
        ret = term
    return ret


def asymptotic_simplify_relation(rel, x):
    """Recursively drop additive terms on both sides of the relation.

    :param rel: relation
    :param x: variable going to infinity
    :returns: a simpler relation with the same asymptotic behaviour

    .. note::

        This should be replaced by Sage's asymptotic ring.
    """
    lhs, rhs = rel.operands()
    op = rel.operator()

    lhs = asymptotic_simplify_term(lhs, x)
    rhs = asymptotic_simplify_term(rhs, x)

    return op(lhs, rhs)
