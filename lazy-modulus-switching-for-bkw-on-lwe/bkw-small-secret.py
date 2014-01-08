# -*- coding: utf-8 -*-
"""
Parameter estimation LWE instances with short secrets
"""
import sys
import pylab

def binary_search_minimize(f, start, stop, **kwds):
    while True:
        x = start + 1*(stop-start)/4
        y = start + 3*(stop-start)/4
        low  = f(x, **kwds)
        high = f(y, **kwds)
        #print "%s, %s :: (%s, %s) vs. (%s, %s)"%(start, stop, x, low, y, high)
        if low <= high:
            start,stop = start, (start+stop)/2
        else:
            start,stop = (start+stop)/2, stop
        if (stop-start) <= 1:
            return start

class BKW:
    def __init__(self, n, t=2.0, q=None, stddev=None, prec=None):
        """
        INPUT:

        - ``n``            - the number of variables in the LWE instance
        - ``t``            - the target number of additions is t*log2(n)
        - ``q``            - the size of the finite field (default: n^2)
        - ``stddev``       - the standard deviation of the LWE instance
        - ``prec``         - bit precision used for numerical computations (default: 2*n)
        """
        self.prec = max(53, n) if prec is None else prec
        RR        = RealField(self.prec)
        self.RR   = RR

        self.a = self.RR(t*log((n),2)) # the target number of additions: a = t*log_2(n)
        self.b = n/self.a # window width b = n/a

        q      = ZZ(q)
        stddev = RR(stddev)
        sigma  = RR(n**t).sqrt() * stddev # after n^t additions we get this stddev

        self.q      = q
        self.stddev = stddev
        self.sigma  = sigma

        self.n = n
        self.t = t

        self.K = GF(self.q)

    @cached_method
    def m(self, success=0.99999):
        RR = self.RR
        vs = self.sigma * RR(sqrt(2*pi)) # ||v||*s = sqrt(n^t)*stddev*sqrt(2*pi)
        adv = RR(exp(-RR(pi)*(RR(vs/self.q)**2)))
        return RR(success)/RR(adv)

    def __repr__(self):
        return "n: %3d, q: %6d, sigma: %.2f, sqrt(2^a)*sigma: %.10f"%(self.n, self.q, self.stddev, self.sigma)

    def __call__(self, success=0.99, latex_output=False):
        """
        Estimate complexity

        INPUT:

        - ``success`` - target total success probability (default: ``0.99``)
        - ``latex_output`` - print as LaTeX table row (default: ``False``)
        """
        RR = self.RR

        q,b,a,n = self.q, self.b, self.a, self.n

        success = self.RR(success)

        m = self.m(success=success)

        stage1a = RR(q**b-1)/RR(2) * ( a*(a-1)/RR(2) * (n+1) - b*a*(a-1)/RR(4) - b/RR(6) * RR( (a-1)**3 + 3/2*(a-1)**2 + 1/RR(2)*(a-1) ) )
        stage1b = m * (a/2 * (n + 2))

        stage1  = stage1a + stage1b

        nrops = stage1

        nbops = log(q,2) * nrops

        ncalls = a * ceil(RR(q**b)/RR(2)) + m

        nmem = ceil(RR(q**b)/RR(2)) * a * (n + 1 - b * (a-1)/2)

        if get_verbose() >= 1:
            print "n: %4d, t: %3.1f,"%(self.n, self.t),
            print "log_2(m): %7.2f,"%(log(m,2).n()),
            print "log_2(#total): %7.2f,"%(log(nrops,2).n()),
            print "log_2(#bops): %7.2f,"%(log(nbops,2).n()),
            print "log_2(#mem): %7.2f,"%(log(nmem,2).n()),
            print "log_2(#Ldis): %7.2f"%(log(ncalls,2).n())

        return self.t, log(m,2).n(), log(nbops,2).n(), log(nmem,2).n(), log(ncalls,2).n()

class BKWSmallSecret:
    def __init__(self, n, q=None, stddev=None, t=2.0, secret_variance=0.25, o=0):
        """
        INPUT:

        - ``n``                    - the number of variables in the LWE instance
        - ``t``                    - the target number of additions is t*log2(n)
        - ``stddev``               - the standard deviation of the LWE instance
        - ``q``                    - the size of the finite field (default: n^2)
        - ``secret_variance``      - variance of secret elements
        """
        q         = ZZ(q)
        self.prec = max(53, 4*n)
        RR        = RealField(self.prec)
        self.RR   = RR
        stddev    = RR(stddev)

        n = ZZ(n)

        self.t = RR(t)
        self.a = self.RR(t*log(n,2)) # the target number of additions: a = t*log_2(n)
        self.b = n/self.a # window width b = n/a

        sigma = RR(n**t).sqrt() * stddev # after n^t additions we get this stddev
        if sigma > 4*q:
            raise ValueError("Precison too low for such noise levels.")

        self.q      = q
        self.stddev = self.RR(stddev)
        self.sigma  = sigma

        self.secret_variance = RR(secret_variance)

        self.n = n
        self.t = t
        self.o = o

        self.K = GF(self.q)

    @cached_method
    def m(self, other_sigma=None):
        RR = self.RR
        success = RR(0.999999)

        if other_sigma is None:
            sigma = self.sigma
        else:
            sigma = RR(sqrt(self.sigma**2 + other_sigma**2))

        vs =  sigma * RR(sqrt(2*pi)) # ||v||*s = sqrt(n^t)*stddev*sqrt(2*pi)
        adv = e**(-RR(pi)*(RR(vs/self.q)**2))
        return RR(success)/adv

    def __repr__(self):
        log2q = self.RR(log(self.q,2))
        return "n: %3d, q: %6d ≈ 2^%.1f, sigma: %.10f"%(self.n, self.q, log2q, self.sigma)


    @staticmethod
    def variance_matrix(q, a, b, kappa, o, RR=None):
        if RR is None:
            RR = RealField()
        q = RR(q)
        a = RR(a).round()
        b = RR(b)
        n = a*b
        kappa = RR(kappa)
        T = RR(2)**(b*kappa)
        n = RR(o)/RR(T*(a+1)) + RR(1)

        U_Var = lambda x: (x**2 - 1)/12
        log2q     = RR(ceil(log(q,2.0)))
        red_var   = 2*U_Var(q/(2**kappa))

        c_ = map(RR, [0.0000000000000000,    
                      0.4057993538687922,    0.6924478992819291,    0.7898852691349439,    0.8441959360364506,    
                      0.8549679124679972,    0.8954469872316165,    0.9157093365103325,    0.9567635780119543, 
                      0.9434245442818547,    0.9987153221343770]);

        M = Matrix(RR, a, a) # rows are tables, columns are entries those tables
        for l in range(M.ncols()):
            for c in range(l, M.ncols()):
                M[l,c] = U_Var(q)

        for l in range(1, a):
            for i in range(0,l):
                M[l,i] = red_var + sum(M[i+1:l].column(i))

            bl = b*l
            if round(bl) < len(c_):
                c_tau = c_[round(bl)]
            else:
                c_tau = RR(1)/RR(5)*RR(sqrt(bl)) + RR(1)/RR(3)

            f = (c_tau*n**(~bl) + 1 - c_tau)**2
            for i in range(l):
                M[l,i] = M[l,i]/f

        if get_verbose() >= 2:
            print M.apply_map(lambda x: log(x,2)).change_ring(RealField(14)).str()

        v = vector(RR, a)
        for i in range(0,a):
            v[i] = red_var + sum(M[i+1:].column(i))
        return M, v

    @cached_method
    def sigma2(self, kappa):
        M, v = BKWSmallSecret.variance_matrix(self.q, self.a, self.b, kappa, self.o, RR=self.RR)
        return sum([self.b * e * self.secret_variance for e in v],RR(0)).sqrt()

    def T(self, kappa):
        return min( self.q**self.b , ZZ(2)**(self.b*kappa) )/2

    def ops_t_e(self, kappa):
        T = self.T(kappa)
        a, n, b = self.a, self.n, self.b

        return T * ( a*(a-1)/2 * (n+1) - b*a*(a-1)/4 - b/6 * ( (a-1)**3 + 3/2*(a-1)**2 + 1/RR(2)*(a-1) ) )

    def ops_m_e(self, m):
        a, n, o = self.a, self.n, self.o
        return  (m + o)  * (a/2 * (n + 2))

    def ops_e(self, kappa, m=None):
        if m is None:
            sigma2 = self.sigma2(kappa)
            m = self.m(sigma2)
            
        return self.ops_t_e(kappa) + self.ops_m_e(m)

    def __call__(self, latex=False):
        if get_verbose() >= 1:
            print self
        n, q, b = map(self.RR, (self.n, self.q, self.b))

        transformation_noise = sqrt(n * 1/RR(12) * self.secret_variance)
        kappa = ceil(log(round(q*transformation_noise/self.stddev),2.0)) + 1

        if kappa > ceil(log(q,2)):
            kappa = ceil(log(q,2))

        a = self.RR(self.a)
        o = self.RR(self.o) if self.o is not None else 0


        best = kappa, RR(2)**(2*n)

        prev_logm = 1.0

        while kappa > 0:
            T = self.T(kappa)
            t = min( round(self.o/T/b), a-1 )
            ops_e = self.ops_e(kappa)
            logm = log(self.m(self.sigma2(kappa=kappa)),2.0)
            if get_verbose() >= 1:
                print "κ: %3d, log(ops_e): %7.2f, m: %7.2f"%(kappa, log(ops_e,2.0), logm)
            # if logm < prev_logm:
            #     kappa +=1
            #     print self.sigma2(kappa=kappa)
            #     break
            prev_logm = logm
            if  ops_e < best[1]:
                best = kappa, ops_e
            elif ops_e >= best[1]:
                break
            kappa -= 1

        kappa = best[0]

        T       = self.T(kappa)
        t       = min( floor(self.o/T/b), a-1 )
        o       = self.o
        m = self.m(self.sigma2(kappa=kappa))
        sigma2 = self.sigma2(kappa=kappa)
        ops_t_e = self.ops_t_e(kappa)
        ops_m_e = self.ops_m_e(m)
        ops_e   = ops_t_e + ops_m_e
        L_e     = T * a + m + o
        mem_e   = T * a * (n + 1 - b * (a-1)/2)

        if get_verbose() >= 1:
            if latex is False:
                print "              t: %7.2f"%self.t
                print "              κ: %7.1f"%(kappa)
                print "              a: %7.1f"%self.a
                print "              b: %7.1f"%self.b
                print "    sqrt(2^a)·σ: 2^%.2f"%log(self.sigma,2.)
                print "             σ2: 2^%.2f"%log(sigma2,2.)
                print "      E[log(m)]: %7.1f"%log(m,2)
                print "         log(o): %7.1f"%(log(o,2) if o else -Infinity)
                print "        E[#ops]: 2^%.2f = 2^%.2f + 2^%.2f"%(log(ops_e,2), log(ops_t_e,2.0), log(ops_m_e,2.0))
                print "       E[#bops]: 2^%.2f"%(log(ops_e*log(q,2),2))
                print "          E[#L]: 2^%.2f"%log(L_e,2)
                print "        E[#mem]: 2^%.2f"%(log(mem_e,2))

            if latex:
                #          n             sigma   kappa a   m       o       ops    L       mem
                print " & %3d &  &  & %5.1f & %2d & %3d & %5.1f & %5.1f & %5.1f & %5.1f & %5.1f & $(2^{%5.1f}, )$\\\\"%(self.n, log(self.stddev,2.0), k, a, log(m,2), log(o,2), log(ops_e,2), log(L_e, 2), log(mem_e, 2), log(ops_e,2))
        return kappa, log(ops_e*log(q,2), 2).n(), log(L_e,2).n(), float(log(m,2)), float(log(mem_e, 2))

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

def focs11_transform(n, q, stddev, s_var):
    transformation_noise = sqrt(n * 1/RR(12) * s_var)
    p = round(q*transformation_noise/stddev)

    beta = ZZ(p)/ZZ(q) * stddev + RR(0.5)  + transformation_noise
    return n, p, beta

def bkw_small_secret_complexity(n, q, stddev, o=None, secret_variance=0.25):
    t = 2*(log(q,2) - log(stddev, 2))/log(n, 2)
    kwds = {"secret_variance":secret_variance, "n":n, "q":q, "stddev":stddev}

    if o is None:
        kappa, logops, logL, logm, logmem, = BKWSmallSecret(t=t, o=0, **kwds)()
        best = kappa, logops, logL, logm, logmem, 0

        for logo in range(floor(logL)-10,floor(logL)+1):
            kappa, logops, logL, logm, logmem, = BKWSmallSecret(t=t, o=2**logo, **kwds)()
            if logops < best[1]:
                best = kappa, logops, logL, logm, logmem, logo

    else:
        kappa, logops, logL, logm, logmem, = BKWSmallSecret(t=t, o=o, **kwds)()
        best = kappa, logops, logL, logm, logmem, float(log(o,2.0))

    kappa, logops, logL, logm, logmem, o = best


    print "%4d & %5.2f & %2d & %4d & %8.1f & %8.1f & %8.1f\\\\ %% %8.1f"%(n, t, kappa, o, logL, logops, logmem, logm)
    sys.stdout.flush()

    return t.n(), o, kappa, logops, logL, logm, logmem

def estimate_complexity(n, small=False, secret_variance=0.25):
    q = previous_prime(ZZ(2)**64)
    stddev = RR(2)**50
    secret_variance = secret_variance

    if small:
        n, q, stddev = focs11_transform(n, q, stddev, secret_variance)
        q = previous_prime(round(q))

    print "%4d &"%n,

    eps, logm, logTq, logops = bkz_complexity(n, q, stddev)
    print "%6d & %8.1f & %8.1f &"%(eps, logm, logops),

    eps, logm, logTq, logops = bkz_complexity(n, q, stddev, bkz2=True)
    print "%6d & %8.1f & %8.1f &"%(eps, logm, logops),
    sys.stdout.flush()
    
    t = RR(2*(log(q,2) - log(stddev, 2))/log(n, 2))
    t, logm, logops, logmem, logL = BKW(n=n, q=q, stddev=stddev, t=t)()
    best = t, logm, logops, logmem, logL
    for t in srange(t+0.1,t+2.15,0.1):
        t, logm, logops, logmem, logL = BKW(n=n, q=q, stddev=stddev, t=t)()
        if logops < best[2]:
            best = t, logm, logops, logmem, logL

    t, logm, logops, logmem, logL = best
    print "%5.2f & %8.1f & %8.1f & %8.1f \\\\"%(t, logL, logops, logmem)
    sys.stdout.flush()

    # t, o, kappa, logops, logL, logm, logmem = bkw_small_secret_complexity(n=n, q=q, stddev=stddev)
    # print "%5.2f & %2d & %4d & %8.1f & %8.1f & %8.1f\\\\ %% %8.1f"%(t, kappa, o, logL, logops, logmem, logm)
    # sys.stdout.flush()q

def estimate_complexity_regev(n, small=False, secret_variance=0.25):
    lwe = Regev(n=n)
    q = lwe.K.order()
    stddev = lwe.D.stddev

    if small:
        n, q, stddev = focs11_transform(n, q, stddev, secret_variance)
        q = previous_prime(round(q))

    print "%4d &"%n,

    eps, logm, logTq, logops = bkz_complexity(n, q, stddev)
    print "%6d & %8.1f & %8.1f &"%(eps, logm, logops),

    eps, logm, logTq, logops = bkz_complexity(n, q, stddev, bkz2=True)
    print "%6d & %8.1f & %8.1f &"%(eps, logm, logops),
    sys.stdout.flush()
    
    t = RR(2*(log(q,2) - log(stddev, 2))/log(n, 2))
    t, logm, logops, logmem, logL = BKW(n=n, q=q, stddev=stddev, t=t)()
    best = t, logm, logops, logmem, logL
    for t in srange(t+0.1,t+2.15,0.1):
        t, logm, logops, logmem, logL = BKW(n=n, q=q, stddev=stddev, t=t)()
        if logops < best[2]:
            best = t, logm, logops, logmem, logL

    t, logm, logops, logmem, logL = best
    print "%5.2f & %8.1f & %8.1f & %8.1f \\\\"%(t, logL, logops, logmem)
    sys.stdout.flush()


def make_plots():
    for b in (1,128):
        h = [(x,l[b][0][1]/y) for x,y in l[b]]
        a,c,x = var('a,c,x')
        model = a*x**(ZZ(1)/b) + c
        model = model.function(x)
        s = find_fit(h[:128],model, solution_dict=True)
        f = model.subs(s)
        #print f
        print s[a], (1-s[a])
        G = line(h) + plot(f, x, 1, h[-1][0], color='green')
        for i,(x_,y_) in enumerate(h):
            if i%4 != 0:
                continue
            print "(%3d, %.4f)"%(x_,y_),

        print
        print
        for i in range(1,512,4):
            print "(%3d, %.4f)"%(i, s[a]*RR(i)**(1/RR(b)) + (1-s[a])),
        print
        print
        print
        print


def make_plots():
    for i,b in enumerate(range(1,128+1)):
        if i%8 == 0:
            print
        h = [(x,l[b][0][1]/y) for x,y in l[b]]
        a,c,x = var('a,c,x')
        model = a*x**(ZZ(1)/b) + c
        model = model.function(x)
        s = find_fit(h[:128],model, solution_dict=True)
        #f = model.subs(s)
        #print f
        print "%20.16f, "%s[a], #(1-s[a])
        # G = line(h) + plot(f, x, 1, h[-1][0], color='green')
        # for i,(x_,y_) in enumerate(h):
        #     if i%4 != 0:
        #         continue
        #     print "(%3d, %.4f)"%(x_,y_),

        # print
        # print
        # for i in range(1,512,4):
        #     print "(%3d, %.4f)"%(i, s[a]*RR(i)**(1/RR(b)) + (1-s[a])),
        # print
        # print
        # print
        # print

def make_plots2():
    interp = [(0,0)]
    for b in range(1,len(l)):
        h = [(x,l[b][0][1]/y) for x,y in l[b]]
        a,c,x = var('a,c,x')
        model = a*x**(ZZ(1)/b) + c
        model = model.function(x)
        s = find_fit(h[:128],model, solution_dict=True)
        interp.append( (b, s[a]) )

    a,c,x = var('a,c,x')
    model = (a*sqrt(x) + c).function(x)
    f = model.subs(find_fit(interp, model, solution_dict=True))
    show(line(interp) + plot(f, 1, interp[-1][0], color='green'))
    print f
