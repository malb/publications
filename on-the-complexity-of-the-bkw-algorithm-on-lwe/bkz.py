
def bkz_complexity(n, q, stddev, force_i=None, bkz2=False):
    RR = RealField(prec=n)
    s = RR(stddev * sqrt(2*pi))

    min_total_cost = None
    params = ()

    for i in range(1, n):
        if force_i is not None:
            i = force_i
        eps = RR(2**-i)
        beta = RR(q/s * sqrt(log(1/eps)/pi))
        delta = RR( 2**( (log(beta,2)**2) / (4*n*log(q,2) ) ) ) 
        m = sqrt(n * log(q, 2)/log(delta,2))
        if get_verbose() >= 2:
            print " delta: %.3f, m: %.1f"%(delta, m)
        m = RR(m * RR(2.0)**i)

        if bkz2 is False:
            log2T_s = RR( 1.8/log(delta,2) - 110 )
        else:
            log2T_s = RR( 0.009/log(delta,2)**2 - 27 )
        log2T_c = RR( log2T_s + log(2.3*10**9,2) + log(1/eps, 2) )
        log2T_q = RR( log2T_c - log(log(q,2),2) )

        if (min_total_cost is None) or (log2T_q < min_total_cost):
            min_total_cost = log2T_q
            params = (-i, log(m,2), log2T_q, log2T_c)

        if force_i is not None:
            break

    if get_verbose() >= 1:
        print "%2d & %6.2f & %6.2f & %6.2f & %6.2f"%(params[0], params[1], params[2], params[3], params[1])
    return params
