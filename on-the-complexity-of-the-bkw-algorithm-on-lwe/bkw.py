"""
Parameter estimation for the BKW algorithm on LWE instances.
"""
import sys
import pylab

class BKW:
    def __init__(self, n, t=2.0, q=None, alpha=None, d=2, prec=None, small_secret=False):
        """
        INPUT:

        - ``n``            - the number of variables in the LWE instance
        - ``t``            - the target number of additions is t*log2(n)
        - ``q``            - the size of the finite field (default: n^2)
        - ``alpha``        - the standard deviation of the LWE instance is alpha*q (default: 1/(sqrt(n)*log^2(n)))
        - ``d``            - the number of elements targeted in hypothesis testing (default: 2)
        - ``prec``         - bit precision used for numerical computations (default: 2*n)
        - ``small_secret`` - apply small secret transformation? (default: False)
        """
        self.prec = 2*n if prec is None else prec
        RR        = RealField(self.prec)
        self.RR   = RR

        self.a = self.RR(t*log((n),2)) # the target number of additions: a = t*log_2(n)
        self.b = n/self.a # window width b = n/a

        q     = ZZ(next_prime(n**2)) if q is None else ZZ(q)
        alpha = 1/(RR(n).sqrt() * RR(log(n,2)**2)) if alpha is None else RR(alpha)
        sigma = RR(n**t).sqrt() * alpha*q # after n^t additions we get this stddev

        self.q     = q
        self.alpha = alpha
        self.sigma = sigma

        self.search_space = ZZ(ceil(2*5*alpha*q+1)) if small_secret else q
        self.small_secret = small_secret

        self.n = n
        self.d = d
        self.t = t

        # we cache some constants that we use often
        self.lower_bound = -floor(q/2)
        self.upper_bound =  floor(q/2)
        self.one_half    =  self.RR(0.5)

        self.K = GF(self.q)

    def cdf(self, x, sigma, mu=0):
        # calling erf() on RealFieldElement is faster than the global erf() function
        try:
            return self.one_half * (1 + ( (x-mu) / (2 * sigma**2 ).sqrt() ).erf() )
        except AttributeError:
            return self.one_half * (1 + erf( (x-mu) / (2 * sigma**2 ).sqrt() ) )

    def compute_p(self, q, sigma):
        oh = self.one_half

        p = {}
        for j in xrange(self.lower_bound, self.upper_bound+1):
            p[j] = self.cdf(j+oh, sigma).n(prec=self.prec) - self.cdf(j-oh, sigma).n(prec=self.prec)

        old_sum = sum(p.values())
        for i in range(1,self.n**2):
            for j in range(self.lower_bound, self.upper_bound+1):
                p[j] += ( self.cdf(  q*i +j + oh, sigma).n(prec=self.prec) - self.cdf(  q*i + j - oh, sigma) ).n(prec=self.prec)
                p[j] += ( self.cdf( -q*i +j + oh, sigma).n(prec=self.prec) - self.cdf( -q*i + j - oh, sigma) ).n(prec=self.prec)
            if get_verbose() >= 1:
                print("i: %3d, sum(p) = %s"%(i, sum(p.values()).str()))
                sys.stdout.flush()
            if  sum(p.values()) == old_sum:
                break
            old_sum = sum(p.values())
        return p

    def m(self, m=None, success=0.99999):
        d = self.d
        # if get_verbose() >= 1:
        #     print "p_success", success
        p = self.p
        q = self.q

        w = {}
        r = {}
        E_c, E_w, V_c, V_w = self.RR(0), self.RR(0), self.RR(0), self.RR(0)

        for j in range(self.lower_bound, self.upper_bound+1):
            r[j] = (q**(d-1) - p[j])/(q**d - 1)
            w[j] = (log(p[j], 2) - log(r[j], 2)).n(prec=self.prec)
            E_c += w[j] * p[j]
            E_w += w[j] * r[j]

        for j in range(self.lower_bound, self.upper_bound+1):
            V_c += p[j] * (w[j] - E_c)**2
            V_w += r[j] * (w[j] - E_w)**2

        from mpmath import mp
        power = int(self.search_space**d - 1)
        if get_verbose() >= 1:
            print "exponent:", power
        mp.prec = self.prec

        def make_f(m):
            return lambda x: (0.5 * (1 + mp.erf( (x - E_w)/mp.sqrt(2*V_w/m) )))**power * 1/(mp.sqrt(2 * mp.pi * V_c/m)) * mp.exp( -(x-E_c)**2/(2*V_c/m) )

        if m is None:
            m = 2
            f = make_f(m)
            intgrl = mp.quad(f, [-mp.inf, mp.inf])
            while intgrl < success:
                m *= 10
                f = make_f(m)
                intgrl = mp.quad(f, [-mp.inf, mp.inf])
                if get_verbose() >= 1:
                    print "log_2(m): %6.2f, p_success: %s"%(log(m,2),intgrl)
                if m > 2**self.n:
                    raise OverflowError
            while intgrl > success:
                m /= 2.0
                f = make_f(m)
                intgrl = mp.quad(f, [-mp.inf, mp.inf])
                if get_verbose() >= 1:
                    print "log_2(m): %6.2f, p_success: %s"%(log(m,2),intgrl)
            m *= 2.0
        else:
            f = make_f(m)
            if get_verbose() >= 1:
                print "log_2(m): %6.2f, p_success: %s"%(log(m,2),mp.quad(f, [-mp.inf, mp.inf]))

        V_w = V_w/m
        V_c = V_c/m

        if get_verbose() >= 1:
            print(" E_c: %.15f,  V_c: %.24f,  E_w: %.15f,  V_w: %.24f"%(E_c, V_c, E_w, V_w))

        return m

    def __repr__(self):
        return "n: %3d, q: %6d, alpha: %.10f, sigma: %.10f"%(self.n, self.q, self.alpha, self.sigma)


    def __getattr__(self, name):
        if name == "p":
            self.p = self.compute_p(self.q, self.sigma)
            return self.p
        else:
            raise AttributeError("Name '%s' unknown."%name)

    def __call__(self, success=0.99, latex_output=False):
        """
        Estimate complexity

        INPUT:

        - ``success`` - target total success probability (default: ``0.99``)
        - ``latex_output`` - print as LaTeX table row (default: ``False``)
        """
        d = self.d
        q = self.q
        b = self.b
        a = self.a
        n = self.n

        repeat = ZZ(ceil(n/d))

        success = self.RR(success)

        m = ZZ(round(self.m(success = success**(1/repeat) )))

        corrector = (q**d)/(q**d - 1)

        stage1a = (q**b-1)/2.0 * ( a*(a-1)/2.0 * (n+1) - b*a*(a-1)/4.0 - b/6.0 * ( (a-1)**3 + 3/2.0*(a-1)**2 + 1/2.0*(a-1) ) )
        stage1b = corrector * (repeat + 1)/2.0 * m * (a/2.0 * (n + 2))
        stage1s = (2*n+1) * (a * ceil(q**b/2.0) +  corrector * repeat * m) if self.small_secret else 0

        stage1  = stage1a + stage1b + stage1s
        stage2  = repeat * m * q**d
        stage3  = (repeat + 1) * d * a * ceil(q**b/2.0)

        nrops = stage1 + stage2 + stage3
        nbops = log(q,2)**2 * nrops

        ncalls = a * ceil(q**b/2.0) +  corrector * repeat * m

        nmem = ceil(q**b/2.0) * a * (n + 1 - b * (a-1)/2)

        if not latex_output:
            print "n: %3d, t: %3.1f, d: %2d,"%(self.n, self.t, d),
            print "log_2(m): %6.2f,"%(log(m,2).n()),
            print "log_2(#sample): %6.2f,"%(log(stage1,2).n()),
            print "log_2(#hypothesis): %6.2f,"%(log(stage2,2).n()),
            print "log_2(#back): %6.2f,"%(log(stage3,2).n()),
            print "log_2(#total): %6.2f,"%(log(nrops,2).n()),
            print "log_2(#bops): %6.2f,"%(log(nbops,2).n()),
            print "log_2(#mem): %6.2f,"%(log(nmem,2).n()),
            print "log_2(#Ldis): %6.2f"%(log(ncalls,2).n())
        else:
            print "%3d &"%(self.n,),
            print "%6.2f &"%(log(m,2).n()),
            print "%6.2f &"%(log(stage1,2).n()),
            print "%6.2f &"%(log(stage2,2).n()),
            print "%6.2f &"%(log(stage3,2).n()),
            print "%6.2f &"%(log(nrops,2).n()),
            print "%6.2f &"%(log(nbops,2).n()),
            print "%6.2f\\\\"%(log(ncalls,2).n())


    def print_parameters(self, success=0.99):
        """
        Print command line for BKW implementation.
        """
        repeat = ZZ(ceil(self.n/self.d))
        m = self.m(success = success**(1/repeat))
        print "./small -n %2d -q %4d -s %.2f -w %2d -m %.2f -v"%(self.n, self.q, self.alpha*self.q, ceil(self.b * log(self.q,2.0).n()), log(m,2.0))


# Micciancio & Regev 2009

pqc_params = [
    (136, 2.1,  2003, 0.0065   ),
    (166, 2.4,  4093, 0.0024   ),
    (192, 2.7,  8191, 0.0009959),
    (214, 3.0, 16381, 0.00045  ),
    (233, 3.2, 32749, 0.000217 )]

# Albrecht et al. 2011

pollycracker_params = [
    ( 136, 2.2,    1999,  0.00558254200346408),
    ( 231, 3.4,   92893, 0.000139563550086602),
    ( 153, 2.4,   12227,  0.00279740858078175),
    ( 253, 3.8,  594397,0.0000349676072597719),]

def call_on_params(p, success=0.99, d=2):
    n,q,alpha,t = p
    BKW(n=n, q=q, alpha=alpha,t=t,d=d)(success=success, latex_output=True)




class BKWDist:
    def __init__(self, n, t=2.0, q=None, alpha=None, prec=None, small_secret=False):
        """
        INPUT:

        - ``n``            - the number of variables in the LWE instance
        - ``t``            - the target number of additions is t*log2(n)
        - ``q``            - the size of the finite field (default: n^2)
        - ``alpha``        - the standard deviation of the LWE instance is alpha*q (default: 1/(sqrt(n)*log^2(n)))
        - ``prec``         - bit precision used for numerical computations (default: 2*n)
        - ``small_secret`` - apply small secret transformation? (default: False)
        """
        self.prec = 2*n if prec is None else prec
        RR        = RealField(self.prec)
        self.RR   = RR

        self.a = self.RR(t*log((n),2)) # the target number of additions: a = t*log_2(n)
        self.b = n/self.a # window width b = n/a

        q     = ZZ(next_prime(n**2)) if q is None else ZZ(q)
        alpha = 1/(RR(n).sqrt() * RR(log(n,2)**2)) if alpha is None else RR(alpha)
        sigma = RR(n**t).sqrt() * alpha*q # after n^t additions we get this stddev

        self.q     = q
        self.alpha = alpha
        self.sigma = sigma

        self.search_space = ZZ(ceil(2*5*alpha*q+1)) if small_secret else q
        self.small_secret = small_secret

        self.n = n
        self.t = t

        # we cache some constants that we use often
        self.lower_bound = -floor(q/2)
        self.upper_bound =  floor(q/2)
        self.one_half    =  self.RR(0.5)

        self.K = GF(self.q)

    def cdf(self, x, sigma, mu=0):
        # calling erf() on RealFieldElement is faster than the global erf() function
        try:
            return self.one_half * (1 + ( (x-mu) / (2 * sigma**2 ).sqrt() ).erf() )
        except AttributeError:
            return self.one_half * (1 + erf( (x-mu) / (2 * sigma**2 ).sqrt() ) )

    def compute_p(self, q, sigma):
        oh = self.one_half

        p = {}
        for j in xrange(self.lower_bound, self.upper_bound+1):
            p[j] = self.cdf(j+oh, sigma).n(prec=self.prec) - self.cdf(j-oh, sigma).n(prec=self.prec)

        old_sum = sum(p.values())
        for i in range(1,self.n**2):
            for j in range(self.lower_bound, self.upper_bound+1):
                p[j] += ( self.cdf(  q*i +j + oh, sigma).n(prec=self.prec) - self.cdf(  q*i + j - oh, sigma) ).n(prec=self.prec)
                p[j] += ( self.cdf( -q*i +j + oh, sigma).n(prec=self.prec) - self.cdf( -q*i + j - oh, sigma) ).n(prec=self.prec)
            if get_verbose() >= 1:
                print("i: %3d, sum(p) = %s"%(i, sum(p.values()).str()))
                sys.stdout.flush()
            if  sum(p.values()) == old_sum:
                break
            old_sum = sum(p.values())
        return p

    def m(self, m=None, success=0.99999):
        # if get_verbose() >= 1:
        #     print "p_success", success

        RR = self.RR

        vs = RR(self.sigma*sqrt(2*pi)) # ||v||*s = sqrt(n^t)*stddev*sqrt(2*pi)


        adv = RR(exp(-pi*(RR(vs/self.q)**2)))

        return success/adv

    def __repr__(self):
        return "n: %3d, q: %6d, alpha: %.10f, sigma: %.10f"%(self.n, self.q, self.alpha, self.sigma)


    def __getattr__(self, name):
        if name == "p":
            self.p = self.compute_p(self.q, self.sigma)
            return self.p
        else:
            raise AttributeError("Name '%s' unknown."%name)

    def __call__(self, success=0.99, latex_output=False):
        """
        Estimate complexity

        INPUT:

        - ``success`` - target total success probability (default: ``0.99``)
        - ``latex_output`` - print as LaTeX table row (default: ``False``)
        """
        RR = self.RR

        q = ZZ(self.q)
        b = RR(self.b)
        a = RR(self.a)
        n = RR(self.n)

        success = self.RR(success)

        m = self.m(success = success )

        stage1a = RR(q**b-1)/RR(2) * ( a*(a-1)/RR(2) * (n+1) - b*a*(a-1)/RR(4) - b/RR(6) * RR( (a-1)**3 + 3/2*(a-1)**2 + 1/2.0*(a-1) ) )
        stage1b = m * (a/2 * (n + 2))
        stage1s = (2*n+1) * (a * (RR(q**b)/2) +  m) if self.small_secret else 0

        stage1  = stage1a + stage1b + stage1s

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

        return self.n, self.t, log(m,2).n(), log(nrops,2).n(prec=self.prec), log(nbops,2).n(), log(nmem,2).n(), log(ncalls,2).n()
