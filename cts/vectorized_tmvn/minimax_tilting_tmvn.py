"""
From https://github.com/brunzema/truncated-mvn-sampler/tree/main
"""
import math
from scipy import optimize
import torch
import time
import numpy as np
from scipy import special
from logging import info, debug, warning

EPS = 10e-15

def torch_erfcinv(x):
    return torch.erfinv(1 - x)

class TruncatedMVN:
    """
    Create a normal distribution :math:`X  \sim N ({\mu}, {\Sigma})` subject to linear inequality constraints
    :math:`lb < X < ub` and sample from it using minimax tilting. Based on the MATLAB implemention by the authors
    (reference below).

    :param torch.ndarray mu: (size D) mean of the normal distribution :math:`\mathbf {\mu}`.
    :param np.ndarray cov: (size D x D) covariance of the normal distribution :math:`\mathbf {\Sigma}`.
    :param np.ndarray lb: (size D) lower bound constrain of the multivariate normal distribution :math:`\mathbf lb`.
    :param np.ndarray ub: (size D) upper bound constrain of the multivariate normal distribution :math:`\mathbf ub`.

    Note that the algorithm may not work if 'cov' is close to being rank deficient.

    Reference:
    Botev, Z. I., (2016), The normal law under linear restrictions: simulation and estimation via minimax tilting,
    Journal of the Royal Statistical Society Series B, 79, issue 1, p. 125-148,

    Example:
        >>> d = 10  # dimensions
        >>>
        >>> # random mu and cov
        >>> mu = np.random.rand(d)
        >>> cov = 0.5 - np.random.rand(d ** 2).reshape((d, d))
        >>> cov = np.triu(cov)
        >>> cov += cov.T - np.diag(cov.diagonal())
        >>> cov = np.dot(cov, cov)
        >>>
        >>> # constraints
        >>> lb = np.zeros_like(mu) - 2
        >>> ub = np.ones_like(mu) * np.inf
        >>>
        >>> # create truncated normal and sample from it
        >>> n_samples = 100000
        >>> samples = TruncatedMVN(mu, cov, lb, ub).sample(n_samples)

    Reimplementation to numpy by Paul Brunzema; Adaptation to torch by Kerrick Johnstonbaugh
    """

    def __init__(self, mu, cov, lb, ub):

        self.tkwargs = {
            "dtype": mu.dtype,
            "device": mu.device
        }        
        self.dim = mu.shape[1]
        
        # called with none when we hack the sampler
        if cov is not None:

            if len(cov.shape) == 2:
                self.dim = len(mu)
                if not cov.shape[0] == cov.shape[1]:
                    raise RuntimeError("Covariance matrix must be of shape DxD!")
                if not (self.dim == cov.shape[0] and self.dim == len(lb) and self.dim == len(ub)):
                    raise RuntimeError("Dimensions D of mean (mu), covariance matric (cov), lower bound (lb) "
                                    "and upper bound (ub) must be the same!")
            elif len(cov.shape) == 3:
                self.dim = mu.shape[1]
                self.n = cov.shape[0] # number of distributions to sample

            self.cov = cov
            self.orig_mu = mu
            self.orig_lb = lb
            self.orig_ub = ub
            
            # permutated
            self.lb = lb - mu  # move distr./bounds to have zero mean
            self.ub = ub - mu  # move distr./bounds to have zero mean
            if torch.any(self.ub <= self.lb):
                raise RuntimeError("Upper bound (ub) must be strictly greater than lower bound (lb) for all D dimensions!")

            # scaled Cholesky with zero diagonal, permutated
            self.L = torch.empty_like(cov)
            self.unscaled_L = torch.empty_like(cov)

            # placeholder for optimization
            self.perm = None
            self.x = None
            self.mu = None
            self.psistar = None

            # for numerics
            self.eps = EPS


    def sqrt(self, x): # helper sqrt to avoid type errors
        return torch.sqrt(torch.tensor(x, **self.tkwargs))



    def vectorized_sample(self, counts):
        """
        Create n samples from the truncated normal distribution.

        :param int n: Number of samples to create.
        :return: D x n array with the samples.
        :rtype: np.ndarray
        """

        # factors (Cholesky, etc.) only need to be computed once!
        if self.psistar is None:
            self.vectorized_compute_factors()

        # start acceptance rejection sampling
        # rv = torch.array([], dtype=torch.float64).reshape(self.dim, 0)

        # first assume we evenly split the samples between the centers

        # rv = torch.zeros((self.n, n // self.n, self.dim), **self.tkwargs)
        if self.n == 1:
            serial_rejection = True
        else:
            serial_rejection = False

        # serial_rejection = True
        # Check if cov is diagonal, if so we can avoid sequential sampling!
        if torch.isclose(self.L, torch.zeros_like(self.L)).all():
            total_samples = torch.sum(counts)
            # max_count = counts.max()
            rv = torch.zeros((self.n, total_samples, self.dim), **self.tkwargs) # be careful, this could be huge
            
            accept, iteration = 0, 0
            n_samples = total_samples
            already_accepted = torch.zeros(self.n, dtype=torch.int64, device=self.tkwargs['device'])
            # TODO: update while condition
            while accept < n_samples:
                mu = self.mu
                psistar = self.psistar
                # logpr, Z = self.vectorized_mvnrnd(n_samples, mu)  # simulate n proposals
                logpr, Z = self.diagonal_mvnrnd(n_samples, mu)
                idx = -torch.log(torch.rand((self.n, n_samples), **self.tkwargs)) > (psistar.unsqueeze(-1) - logpr)  # acceptance tests, idx is a mask
                
                n_accept = idx.sum(dim=1) # number of samples that pass acceptance test per unique center
                
                # TODO: vectorize
                for i in range(self.n):
                    accepted_samples = Z[i].T[idx[i]]
                    start_idx = already_accepted[i]
                    n_accept_i = torch.min(n_accept[i], counts[i]-already_accepted[i]).to(torch.int64)
                    rv[i][start_idx:start_idx+n_accept_i] = accepted_samples[:n_accept_i]  # accumulate accepted
                    already_accepted[i] += n_accept_i

                accept = already_accepted.sum()
                iteration += 1
                if iteration == 10 ** 3:
                    warning('Acceptance prob. smaller than 0.001.')
                elif iteration > 10 ** 4:
                    accept = n_samples
                    rv = torch.concatenate((rv, Z), axis=1)
                    warning('Sample is only approximately distributed.')


            # finish sampling and postprocess the samples!
            # order = self.perm.argsort(axis=0)
            rv = torch.cat([rv[i][:counts[i],:] for i in range(self.n)], dim=0)
            rv = rv.unsqueeze(-1) # batch of column vectors

        else:
            if serial_rejection:        
                rv = [torch.empty((self.dim, 0), **self.tkwargs) for i in range(self.n)]
                for i in range(self.n):
                    accept, iteration = 0, 0
                    n_samples = counts[i]
                    while accept < n_samples:
                        mu = self.mu[i].squeeze()
                        psistar = self.psistar[i].squeeze()
                        logpr, Z = self.serial_mvnrnd(n_samples, mu, self.L[i].squeeze(), self.lb[i].squeeze(), self.ub[i].squeeze())  # simulate n proposals
                        idx = -torch.log(torch.rand(n_samples, **self.tkwargs)) > (psistar - logpr)  # acceptance tests
                        
                        rv[i] = torch.concatenate((rv[i], Z[:, idx]), axis=1)  # accumulate accepted
                        accept = rv[i].shape[1]  # keep track of # of accepted
                        iteration += 1
                        if iteration == 10 ** 3:
                            warning('Acceptance prob. smaller than 0.001.')
                        elif iteration > 10 ** 4:
                            accept = n_samples
                            rv = torch.concatenate((rv, Z), axis=1)
                            warning('Sample is only approximately distributed.')


                # finish sampling and postprocess the samples!
                # order = self.perm.argsort(axis=0)
                
                rv = torch.cat([rv[i][:, :counts[i]] for i in range(self.n)], dim=1)
                rv = rv.T.unsqueeze(-1) # batch of column vectors

            else:
                total_samples = torch.sum(counts)
                # max_count = counts.max()
                rv = torch.zeros((self.n, total_samples, self.dim), **self.tkwargs) # be careful, this could be huge
                
                accept, iteration = 0, 0
                n_samples = total_samples
                already_accepted = torch.zeros(self.n, dtype=torch.int64, device=self.tkwargs['device'])
                # TODO: update while condition
                while accept < n_samples:
                    mu = self.mu
                    psistar = self.psistar
                    logpr, Z = self.vectorized_mvnrnd(n_samples, mu)  # simulate n proposals
                    idx = -torch.log(torch.rand((self.n, n_samples), **self.tkwargs)) > (psistar.unsqueeze(-1) - logpr)  # acceptance tests, idx is a mask
                    
                    n_accept = idx.sum(dim=1) # number of samples that pass acceptance test per unique center
                    # TODO: vectorize
                    
                    for i in range(self.n):
                        accepted_samples = Z[i].T[idx[i]]
                        start_idx = already_accepted[i]
                        n_accept_i = torch.min(n_accept[i], counts[i]-already_accepted[i]).to(torch.int64)
                        rv[i][start_idx:start_idx+n_accept_i] = accepted_samples[:n_accept_i]  # accumulate accepted
                        already_accepted[i] += n_accept_i

                    accept = already_accepted.sum()
                    iteration += 1
                    if iteration == 10 ** 3:
                        warning('Acceptance prob. smaller than 0.001.')
                    elif iteration > 10 ** 4:
                        accept = n_samples
                        rv = torch.concatenate((rv, Z), axis=1)
                        warning('Sample is only approximately distributed.')


                # finish sampling and postprocess the samples!
                # order = self.perm.argsort(axis=0)
                rv = torch.cat([rv[i][:counts[i],:] for i in range(self.n)], dim=0)
                rv = rv.unsqueeze(-1) # batch of column vectors

        return rv

    def sample(self, n):
        """
        Create n samples from the truncated normal distribution.

        :param int n: Number of samples to create.
        :return: D x n array with the samples.
        :rtype: np.ndarray
        """
        start_t = time.perf_counter()
        if not isinstance(n, int):
            raise RuntimeError("Number of samples must be an integer!")

        # factors (Cholesky, etc.) only need to be computed once!
        if self.psistar is None:
            opt_t, decomp_t = self.compute_factors()

        # start acceptance rejection sampling
        # rv = torch.array([], dtype=torch.float64).reshape(self.dim, 0)
        rv = torch.empty((self.dim, 0), **self.tkwargs)
        accept, iteration = 0, 0
        while accept < n:
            logpr, Z = self.mvnrnd(n, self.mu)  # simulate n proposals
            idx = -torch.log(torch.rand(n, **self.tkwargs)) > (self.psistar - logpr)  # acceptance tests
            rv = torch.concatenate((rv, Z[:, idx]), axis=1)  # accumulate accepted
            accept = rv.shape[1]  # keep track of # of accepted
            iteration += 1
            if iteration == 10 ** 3:
                warning('Acceptance prob. smaller than 0.001.')
            elif iteration > 10 ** 4:
                accept = n
                rv = torch.concatenate((rv, Z), axis=1)
                warning('Sample is only approximately distributed.')

        # finish sampling and postprocess the samples!
        order = self.perm.argsort(axis=0)
        rv = rv[:, :n]

        end_t = time.perf_counter()
        total_elapsed = end_t - start_t
        
        debug(f'TMVN sample total elapsed time: {total_elapsed}')
        debug(f'proportion opt: {opt_t/total_elapsed}')
        debug(f'proportion decomp: {decomp_t/total_elapsed}')

        return rv, self.unscaled_L, order
    
    def vectorized_compute_factors(self):
        # compute permutated Cholesky factor and solve optimization

        # Cholesky decomposition of matrix with permuation
        decomp_start_t = time.perf_counter()
        self.unscaled_L, self.perm = self.vectorized_colperm()
        decomp_end_t = time.perf_counter()
        # check that vectorized cholesky decomposition was accurate
        for Sigma, L in zip(self.cov, self.unscaled_L):
            Sigma = Sigma.squeeze(0)
            L = L.squeeze(0)
            assert (torch.isclose(Sigma, L @ L.T)).all()

        D = torch.diagonal(self.unscaled_L, dim1=-2, dim2=-1) # batch diagonal
        if torch.any(D < self.eps):
            warning('Method might fail as covariance matrix is singular!')

        # rescale
        # copy_mat = torch.eye(self.n, **self.tkwargs).unsqueeze(-1).repeat(1,1,self.dim)
        # scale_factor = D.T.unsqueeze(0).repeat(self.n,1,1) @ copy_mat
        scale_factor = D.unsqueeze(-1).repeat(1,1,self.dim)
        # scaled_L = self.unscaled_L / torch.tile(D.reshape(self.dim, 1), (1, self.dim))
        scaled_L = self.unscaled_L / scale_factor
        self.lb = self.lb / D.unsqueeze(-1)
        self.ub = self.ub / D.unsqueeze(-1)

        # remove diagonal
        self.L = scaled_L - torch.eye(self.dim, **self.tkwargs).unsqueeze(0).repeat(self.n, 1, 1)

        opt_start_t = time.perf_counter()
        # get gradient/Jacobian function
        gradpsi_fn = self.get_gradient_function()
        x0 = torch.zeros(2 * (self.dim - 1)).numpy()

        # find optimal tilting parameter non-linear equation solver
        # sol = optimize.root(gradpsi, x0, args=(self.L, self.lb, self.ub), method='hybr', jac=True)
        sol_xs = torch.zeros(self.n, self.dim-1, **self.tkwargs)
        sol_mus = torch.zeros(self.n, self.dim-1, **self.tkwargs)
        for i in range(self.n):
            if self.dim > 1:
                L_i = self.L[i].squeeze(0).cpu().numpy()
                lb_i = self.lb[i].squeeze(-1).cpu().numpy()
                ub_i = self.ub[i].squeeze(-1).cpu().numpy()
            else:
                L_i = self.L[i].cpu().numpy()
                lb_i = self.lb[i].cpu().numpy()
                ub_i = self.ub[i].cpu().numpy()
            sol = optimize.root(gradpsi_fn, x0, args=(L_i, lb_i, ub_i), method='hybr', jac=True) # wrap this function with cache

            # x0 = sol.x # speed up optimization by using previous solution as initial guess
            sol_x = torch.tensor(sol.x, **self.tkwargs)
            sol_xs[i] = sol_x[:self.dim - 1]
            sol_mus[i] = sol_x[self.dim - 1:]
            
            if not sol.success:
                warning('Method may fail as covariance matrix is close to singular!')
        
        opt_end_t = time.perf_counter() 
        self.x = sol_xs
        self.mu = sol_mus
        # compute psi star
        self.psistar = self.psy(self.x, self.mu)

        return opt_end_t - opt_start_t, decomp_end_t - decomp_start_t
    
    def compute_factors(self):
        # compute permutated Cholesky factor and solve optimization

        # Cholesky decomposition of matrix with permuation
        decomp_start_t = time.perf_counter()
        self.unscaled_L, self.perm = self.colperm()
        decomp_end_t = time.perf_counter()

        D = torch.diag(self.unscaled_L)
        if torch.any(D < self.eps):
            warning('Method might fail as covariance matrix is singular!')

        # rescale
        scaled_L = self.unscaled_L / torch.tile(D.reshape(self.dim, 1), (1, self.dim))
        self.lb = self.lb / D
        self.ub = self.ub / D

        # remove diagonal
        self.L = scaled_L - torch.eye(self.dim, **self.tkwargs)

        opt_start_t = time.perf_counter()
        # get gradient/Jacobian function
        gradpsi_fn = self.get_gradient_function()
        x0 = torch.zeros(2 * (self.dim - 1))

        # find optimal tilting parameter non-linear equation solver
        # sol = optimize.root(gradpsi, x0, args=(self.L, self.lb, self.ub), method='hybr', jac=True)
        sol = optimize.root(gradpsi_fn, x0.numpy(), args=(self.L.cpu().numpy(), self.lb.cpu().numpy(), self.ub.cpu().numpy()), method='hybr', jac=True)
        if not sol.success:
            warning('Method may fail as covariance matrix is close to singular!')
        
        opt_end_t = time.perf_counter()
        
        sol_x = torch.tensor(sol.x, **self.tkwargs)
        self.x = sol_x[:self.dim - 1]
        self.mu = sol_x[self.dim - 1:] 

        # compute psi star
        self.psistar = self.psy(self.x, self.mu)

        return opt_end_t - opt_start_t, decomp_end_t - decomp_start_t
        
    def reset(self):
        # reset factors -> when sampling, optimization for optimal tilting parameters is performed again

        # permutated
        self.lb = self.orig_lb - self.orig_mu  # move distr./bounds to have zero mean
        self.ub = self.orig_ub - self.orig_mu

        # scaled Cholesky with zero diagonal, permutated
        self.L = torch.empty_like(self.cov)
        self.unscaled_L = torch.empty_like(self.cov)

        # placeholder for optimization
        self.perm = None
        self.x = None
        self.mu = None
        self.psistar = None

    def serial_mvnrnd(self, n, mu, L, lb, ub):
        # generates the proposals from the exponentially tilted sequential importance sampling pdf
        # output:     logpr, log-likelihood of sample
        #             Z, random sample
        mu = torch.cat([mu, torch.zeros(1, **self.tkwargs)]).to(self.L)
        Z = torch.zeros((self.dim, n), **self.tkwargs)
        logpr = 0
        for k in range(self.dim):
            # compute matrix multiplication L @ Z
            col = L[k, :k] @ Z[:k, :]
            # compute limits of truncation
            tl = lb[k] - mu[k] - col
            tu = ub[k] - mu[k] - col
            # simulate N(mu,1) conditional on [tl,tu]
            Z[k, :] = mu[k] + self.trandn(tl, tu)
            # update likelihood ratio
            logpr += self.lnNormalProb(tl, tu) + .5 * mu[k] ** 2 - mu[k] * Z[k, :]
        return logpr, Z

    def diagonal_mvnrnd(self, n, mu):
        # generates the proposals from the exponentially tilted sequential importance sampling pdf
        # output:     logpr, log-likelihood of sample
        #             Z, random sample

        mu = torch.cat([mu, torch.zeros((self.n, 1), **self.tkwargs)], dim=-1).to(self.L)
        mu = mu.unsqueeze(-1)

        # compute limits of truncation
        tl = (self.lb - mu).repeat(1,1,n)
        tu = (self.ub - mu).repeat(1,1,n)
        # simulate N(mu,1) conditional on [tl,tu]
        trandn_sample = self.trandn(tl, tu)
        Z = mu + trandn_sample
        # update likelihood ratio
        mu_T = mu.transpose(1,2)
        logpr = torch.sum(self.lnNormalProb(tl, tu), dim=1) + 0.5 * (mu_T @ mu).squeeze(-1) - (mu_T @ Z).squeeze()
        
        return logpr.squeeze(), Z

    def vectorized_mvnrnd(self, n, mu):
        # generates the proposals from the exponentially tilted sequential importance sampling pdf
        # output:     logpr, log-likelihood of sample
        #             Z, random sample
        # Z = torch.zeros((self.n, self.dim, n), **self.tkwargs)

        mu = torch.cat([mu, torch.zeros((self.n, 1), **self.tkwargs)], dim=-1).to(self.L)
        Z = torch.zeros((self.n, self.dim, n), **self.tkwargs)
        logpr = torch.zeros(self.n, n, **self.tkwargs)
        

        for k in range(self.dim):
            # compute matrix multiplication L @ Z
            col = self.L[:, k, :k].unsqueeze(1) @ Z[:, :k, :]
            # compute limits of truncation
            tl = self.lb[:, k] - mu[:, k].unsqueeze(-1) - col.squeeze()
            tu = self.ub[:, k] - mu[:, k].unsqueeze(-1) - col.squeeze()
            # simulate N(mu,1) conditional on [tl,tu]
            trandn_sample = self.trandn(tl, tu)
            
            Z[:, k, :] = mu[:, k].unsqueeze(-1) + trandn_sample
            # update likelihood ratio

            logpr += self.lnNormalProb(tl, tu) + .5 * mu[:, k].unsqueeze(-1) ** 2 - mu[:, k].unsqueeze(-1) * Z[:, k, :]

        return logpr, Z

    def mvnrnd(self, n, mu):
        # generates the proposals from the exponentially tilted sequential importance sampling pdf
        # output:     logpr, log-likelihood of sample
        #             Z, random sample
        mu = torch.cat([mu, torch.zeros(1, **self.tkwargs)]).to(self.L)
        Z = torch.zeros((self.dim, n), **self.tkwargs)
        logpr = 0
        # TODO: if unscaled L is diagonal, can avoid loop and compute vals in parallel
        for k in range(self.dim):
            # compute matrix multiplication L @ Z
            col = self.L[k, :k] @ Z[:k, :]
            # compute limits of truncation
            tl = self.lb[k] - mu[k] - col
            tu = self.ub[k] - mu[k] - col
            # simulate N(mu,1) conditional on [tl,tu]
            Z[k, :] = mu[k] + self.trandn(tl, tu)
            # update likelihood ratio
            logpr += self.lnNormalProb(tl, tu) + .5 * mu[k] ** 2 - mu[k] * Z[k, :]
        return logpr, Z

    def trandn(self, lb, ub):
        """
        Sample generator for the truncated standard multivariate normal distribution :math:`X \sim N(0,I)` s.t.
        :math:`lb<X<ub`.

        If you wish to simulate a random variable 'Z' from the non-standard Gaussian :math:`N(m,s^2)`
        conditional on :math:`lb<Z<ub`, then first simulate x=TruncatedMVNSampler.trandn((l-m)/s,(u-m)/s) and set
        Z=m+s*x.
        Infinite values for 'ub' and 'lb' are accepted.

        :param np.ndarray lb: (size D) lower bound constrain of the normal distribution :math:`\mathbf lb`.
        :param np.ndarray ub: (size D) upper bound constrain of the normal distribution :math:`\mathbf lb`.

        :return: D samples if the truncated normal distribition x ~ N(0, I) subject to lb < x < ub.
        :rtype: np.ndarray
        """
        if not len(lb) == len(ub):
            raise RuntimeError("Lower bound (lb) and upper bound (ub) must be of the same length!")

        x = torch.empty_like(lb)
        a = 0.66  # threshold used in MATLAB implementation
        # three cases to consider
        # case 1: a<lb<ub
        I = lb > a
        if torch.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = self.ntail(tl, tu)
        # case 2: lb<ub<-a
        J = ub < -a
        if torch.any(J):
            tl = -ub[J]
            tu = -lb[J]
            x[J] = - self.ntail(tl, tu)
        # case 3: otherwise use inverse transform or accept-reject
        I = ~(I | J)
        if torch.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = self.tn(tl, tu)
        return x

    def tn(self, lb, ub, tol=2):
        # samples a column vector of length=len(lb)=len(ub) from the standard multivariate normal distribution
        # truncated over the region [lb,ub], where -a<lb<ub<a for some 'a' and lb and ub are column vectors
        # uses acceptance rejection and inverse-transform method

        sw = tol  # controls switch between methods, threshold can be tuned for maximum speed for each platform
        x = torch.empty_like(lb)
        # case 1: abs(ub-lb)>tol, uses accept-reject from randn
        I = abs(ub - lb) > sw
        if torch.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = self.trnd(tl, tu)

        # case 2: abs(u-l)<tol, uses inverse-transform
        I = ~I
        if torch.any(I):
            tl = lb[I]
            tu = ub[I]
            pl = torch.special.erfc(tl / self.sqrt(2)) / 2
            pu = torch.special.erfc(tu / self.sqrt(2)) / 2
            # x[I] = np.sqrt(2) * special.erfcinv(2 * (pl - (pl - pu) * self.random_state.rand(len(tl)))) # NOTE: torch does not have efcinv, buf erfcinv(x) = erfinv(1-x) 
            x[I] = self.sqrt(2) * torch_erfcinv(2 * (pl - (pl - pu) * torch.rand(len(tl), **self.tkwargs)))
        return x

    def trnd(self, lb, ub):
        # uses acceptance rejection to simulate from truncated normal
        x = torch.randn(len(lb), **self.tkwargs)  # sample normal
        test = (x < lb) | (x > ub)
        I = torch.where(test)[0]
        d = len(I)
        while d > 0:  # while there are rejections
            ly = lb[I]
            uy = ub[I]
            y = torch.randn(len(uy), **self.tkwargs)  # resample
            idx = (y > ly) & (y < uy)  # accepted
            x[I[idx]] = y[idx]
            I = I[~idx]
            d = len(I)
        return x

    def ntail(self, lb, ub):
        # samples a column vector of length=len(lb)=len(ub) from the standard multivariate normal distribution
        # truncated over the region [lb,ub], where lb>0 and lb and ub are column vectors
        # uses acceptance-rejection from Rayleigh distr. similar to Marsaglia (1964)
        if not len(lb) == len(ub):
            raise RuntimeError("Lower bound (lb) and upper bound (ub) must be of the same length!")
        c = (lb ** 2) / 2
        n = len(lb)
        f = torch.expm1(c - ub ** 2 / 2)
        x = c - torch.log(1 + torch.rand(n, **self.tkwargs) * f)  # sample using Rayleigh
        # keep list of rejected
        I = torch.where(torch.rand(n, **self.tkwargs) ** 2 * x > c)[0]
        d = len(I)
        while d > 0:  # while there are rejections
            cy = c[I]
            y = cy - torch.log(1 + torch.rand(d, **self.tkwargs) * f[I])
            idx = (torch.rand(d, **self.tkwargs) ** 2 * y) < cy  # accepted
            x[I[idx]] = y[idx]  # store the accepted
            I = I[~idx]  # remove accepted from the list
            d = len(I)
        return torch.sqrt(2 * x)  # this Rayleigh transform can be delayed till the end

    def psy(self, x, mu):
        """
        x, mu are n x (d-1)

        """
        # implements psi(x,mu); assumes scaled 'L' without diagonal
        x = torch.cat([x, torch.zeros(self.n, 1, **self.tkwargs)], dim=1).to(self.L)
        mu = torch.cat([mu, torch.zeros(self.n, 1, **self.tkwargs)], dim=1).to(self.L)
        c = self.L @ x.unsqueeze(-1)
        lt = self.lb - mu.unsqueeze(-1) - c
        ut = self.ub - mu.unsqueeze(-1) - c
        # p = torch.sum(self.lnNormalProb(lt, ut) + 0.5 * mu ** 2 - x * mu)
        # TODO: confirm this matches the solution from singular samples
        p = torch.sum(self.lnNormalProb(lt.squeeze(-1), ut.squeeze(-1)) + 0.5 * mu.squeeze(-1) ** 2 - x.squeeze(-1) * mu.squeeze(-1), dim=1)
        # p = torch.sum(self.lnNormalProb(lt.squeeze(), ut.squeeze()) + 0.5 * mu.squeeze() ** 2 - x.squeeze() * mu.squeeze(), dim=1)
        return p

    def get_gradient_function(self, fn_type='numpy'):
        # wrapper to avoid dependancy on self

        def gradpsi(y, L, l, u):
            # implements gradient of psi(x) to find optimal exponential twisting, returns also the Jacobian
            # NOTE: assumes scaled 'L' with zero diagonal
            d = len(u)
            c = np.zeros(d)
            mu, x = c.copy(), c.copy()
            x[0:d - 1] = y[0:d - 1]
            mu[0:d - 1] = y[d - 1:]

            # compute now ~l and ~u
            c[1:d] = L[1:d, :] @ x
            lt = l - mu - c
            ut = u - mu - c

            # compute gradients avoiding catastrophic cancellation
            w = np_lnNormalProb(lt, ut)
            pl = np.exp(-0.5 * lt ** 2 - w) / np.sqrt(2 * math.pi) # lower case p is little psi above equation 8 (normal pdf). Subtracting w in log space accounts for normalization (denominator of Psi above eq 8)?
            pu = np.exp(-0.5 * ut ** 2 - w) / np.sqrt(2 * math.pi)
            P = pl - pu

            # output the gradient
            dfdx = - mu[0:d - 1] + (P.T @ L[:, 0:d - 1]).T
            dfdm = mu - x + P
            grad = np.concatenate((dfdx, dfdm[:-1]), axis=0)

            # construct jacobian
            lt[np.isinf(lt)] = 0
            ut[np.isinf(ut)] = 0

            dP = - P ** 2 + lt * pl - ut * pu # Psi' (Psi prime) from line above eq 8
            DL = np.tile(dP.reshape(d, 1), (1, d)) * L
            mx = DL - np.eye(d)
            xx = L.T @ DL
            mx = mx[:-1, :-1]
            xx = xx[:-1, :-1]
            J = np.block([[xx, mx.T],
                          [mx, np.diag(1 + dP[:-1])]])
            return (grad, J)

        if fn_type == 'numpy':
            return gradpsi

    def colperm(self):
        perm = torch.arange(self.dim)
        L = torch.zeros_like(self.cov)
        z = torch.zeros_like(self.orig_mu)

        for j in torch.arange(self.dim):
            pr = torch.ones_like(z) * torch.inf  # compute marginal prob.
            I = torch.arange(j, self.dim)  # search remaining dimensions
            D = torch.diag(self.cov)
            s = D[I] - torch.sum(L[I, 0:j] ** 2, axis=1)
            s[s < 0] = self.eps
            s = torch.sqrt(s)
            tl = (self.lb[I] - L[I, 0:j] @ z[0:j]) / s # Kerrick note: looks like tilde bounds in separation of variables estimator from Botev pg 3
            tu = (self.ub[I] - L[I, 0:j] @ z[0:j]) / s
            pr[I] = self.lnNormalProb(tl, tu)
            # find smallest marginal dimension
            k = torch.argmin(pr)

            # flip dimensions k-->j
            jk = [int(j), int(k)]
            kj = [int(k), int(j)]
            self.cov[jk, :] = self.cov[kj, :]  # update rows of cov
            self.cov[:, jk] = self.cov[:, kj]  # update cols of cov
            L[jk, :] = L[kj, :]  # update only rows of L
            self.lb[jk] = self.lb[kj]  # update integration limits
            self.ub[jk] = self.ub[kj]  # update integration limits
            perm[jk] = perm[kj]  # keep track of permutation

            # construct L sequentially via Cholesky computation (Kerrick note: looks like Cholesky-Banachiewicz algorithm)
            # s = self.cov[j, j] - torch.sum(L[j, 0:j] ** 2, axis=0)
            s = self.cov[j, j] - L[j, 0:j] @ L[j, 0:j] # inner product to get sum of squared entried
            if s < -0.01:
                raise RuntimeError("Sigma is not positive semi-definite")
            elif s < 0:
                s = self.eps
            L[j, j] = torch.sqrt(s)
            new_L = self.cov[j + 1:self.dim, j] - L[j + 1:self.dim, 0:j] @ L[j, 0:j] # original implementation includes transpose that throws warning
            L[j + 1:self.dim, j] = new_L / L[j, j]

            # find mean value, z(j), of truncated normal
            tl = (self.lb[j] - L[j, 0:j - 1] @ z[0:j - 1]) / L[j, j] # Kerrick note: tilde bounds from SOV estimator
            tu = (self.ub[j] - L[j, 0:j - 1] @ z[0:j - 1]) / L[j, j]
            w = self.lnNormalProb(tl, tu)  # aids in computing expected value of trunc. normal
            z[j] = (torch.exp(-0.5 * tl ** 2 - w) - torch.exp(-0.5 * tu ** 2 - w)) / self.sqrt(2 * math.pi)
        return L, perm

    def vectorized_colperm(self):
        "adapted from colperm function to compute cholesky of multiple covariance matrices in parallel"
        # first test 2 of the same cov matrices in parallel
        n = self.n
        # perm = torch.arange(self.dim).unsqueeze(0).repeat(n, 1)
        perm = torch.eye(self.dim, **self.tkwargs).unsqueeze(0).repeat(n, 1, 1)
        L = torch.zeros_like(self.cov, **self.tkwargs) 
        z = torch.zeros_like(self.orig_mu, **self.tkwargs).unsqueeze(-1)

        # self.cov = self.cov.repeat(n, 1, 1) # tmp repeat cov for debug
        self.lb = self.lb.unsqueeze(-1)
        self.ub = self.ub.unsqueeze(-1)

        # only do cholesky if we dont have a diagonal covariance
        if torch.isclose(self.cov, torch.diag_embed(torch.diagonal(self.cov, dim1=-2, dim2=-1))).all():

            L = torch.sqrt(self.cov)
            tl = self.lb / torch.diagonal(L, dim1=-2, dim2=-1).unsqueeze(-1)
            tu = self.ub / torch.diagonal(L, dim1=-2, dim2=-1).unsqueeze(-1)
            pr = self.lnNormalProb(tl.squeeze(-1), tu.squeeze(-1))
            
            # get the order
            # row_indices = torch.arange(self.dim, device=self.tkwargs['device']).repeat(n,1)
            _, row_indices = torch.sort(pr, dim=1)
            P = torch.zeros((n,self.dim,self.dim), **self.tkwargs).scatter(2, row_indices.unsqueeze(-1), 1)
            
            self.lb = P @ self.lb
            self.ub = P @ self.ub
            self.cov = P @ self.cov @ P.transpose(1,2) # update rows and cols
            L = torch.sqrt(self.cov)
            
            perm = P


        else:
            # perform cholesky with pivots to reduce variance
            for j in torch.arange(self.dim, device=self.tkwargs['device']):
                pr = torch.ones_like(z) * torch.inf  # compute marginal prob.
                # I = torch.arange(j, self.dim) # search remaining dimensions
                D = torch.diagonal(self.cov, dim1=-2, dim2=-1) # batch diagonal
                s = D[:, j:] - torch.sum(L[:, j:, :j] ** 2, axis=2)
                s[s < 0] = self.eps
                s = torch.sqrt(s).unsqueeze(-1)
                tl = (self.lb[:, j:] - L[:, j:, :j] @ z[:, :j]) / s # Kerrick note: looks like tilde bounds in separation of variables estimator from Botev pg 3
                tu = (self.ub[:, j:] - L[:, j:, :j] @ z[:, :j]) / s
                pr[:, j:] = self.lnNormalProb(tl, tu)
                # find smallest marginal dimension
                k = torch.argmin(pr, dim=1)

                # flip dimensions k-->j
                # jk = [int(j), int(k)]
                jk = torch.cat([j.repeat(n,1), k], dim=1).to(torch.int64)
                # kj = [int(k), int(j)]
                kj = torch.cat([k, j.repeat(n,1)], dim=1).to(torch.int64)

                # form batch permuation matrices
                row_indices = torch.arange(self.dim, device=self.tkwargs['device']).repeat(n,1)
                row_indices = row_indices.scatter(1, jk, kj)
                P = torch.zeros((n,self.dim,self.dim), **self.tkwargs).scatter(2, row_indices.unsqueeze(-1), 1)
                # self.cov[:, jk, :] = self.cov[:, kj, :]  # update rows of cov
                # self.cov[:, :, jk] = self.cov[:, :, kj]  # update cols of cov
                self.cov = P @ self.cov @ P.transpose(1,2) # update rows and cols
                # L[:, jk, :] = L[:, kj, :]  # update only rows of L
                L = P @ L
                # self.lb[:, jk] = self.lb[:, kj]  # update integration limits
                # self.ub[:, jk] = self.ub[:, kj]  # update integration limits
                self.lb = P @ self.lb
                self.ub = P @ self.ub
                # perm[:, jk] = perm[:, kj]  # keep track of permutation
                perm = P @ perm

                # construct L sequentially via Cholesky computation (Kerrick note: looks like Cholesky-Banachiewicz algorithm)
                # s = self.cov[j, j] - torch.sum(L[j, 0:j] ** 2, axis=0)
                # s = self.cov[j, j] - L[j, 0:j] @ L[j, 0:j] # inner product to get sum of squared entried
                s = self.cov[:, j, j] - torch.sum(L[:, j, :j] ** 2, axis=1)
                if (s < -0.01).any():
                    raise RuntimeError("Sigma is not positive semi-definite")
                elif (s < 0).any():
                    s[s < 0] = self.eps
                L[:, j, j] = torch.sqrt(s)
                new_L = self.cov[:, j + 1:, j] - (L[:, j + 1:, :j] @ L[:, j, :j].unsqueeze(-1)).squeeze(-1) # original implementation includes transpose that throws warning
                L[:, j + 1:, j] = new_L / L[:, j, j].unsqueeze(1)

                # find mean value, z(j), of truncated normal
                tl = (self.lb[:, j] - (L[:, j, :j - 1].unsqueeze(1) @ z[:, :j - 1]).squeeze(2)) / L[:, j, j].unsqueeze(1) # Kerrick note: tilde bounds from SOV estimator
                tu = (self.ub[:, j] - (L[:, j, :j - 1].unsqueeze(1) @ z[:, :j - 1]).squeeze(2)) / L[:, j, j].unsqueeze(1)
                w = self.lnNormalProb(tl, tu)  # aids in computing expected value of trunc. normal
                z[:, j] = (torch.exp(-0.5 * tl ** 2 - w) - torch.exp(-0.5 * tu ** 2 - w)) / self.sqrt(2 * math.pi)

        # assert torch.isclose(L, diag_L).all()
        # assert torch.isclose(self.lb, diag_lb).all()
        # assert torch.isclose(self.ub, diag_ub).all()
        # assert torch.isclose(self.cov, diag_cov).all()
        # assert torch.isclose(perm, diag_perm).all()

        return L, perm

    def lnNormalProb(self, a, b): # attach to class so we can use self.sqrt for convenience
        # computes ln(P(a<Z<b)) where Z~N(0,1) very accurately for any 'a', 'b'
        p = torch.zeros_like(a)
        # case b>a>0
        I = a > 0
        if torch.any(I):
            pa = self.lnPhi(a[I])
            pb = self.lnPhi(b[I])
            p[I] = pa + torch.log1p(-torch.exp(pb - pa))
        # case a<b<0
        idx = b < 0
        if torch.any(idx):
            pa = self.lnPhi(-a[idx])  # log of lower tail
            pb = self.lnPhi(-b[idx])
            p[idx] = pb + torch.log1p(-torch.exp(pa - pb))
        # case a < 0 < b
        I = (~I) & (~idx)
        if torch.any(I):
            pa = torch.special.erfc(-a[I] / self.sqrt(2)) / 2  # lower tail
            pb = torch.special.erfc(b[I] / self.sqrt(2)) / 2  # upper tail
            p[I] = torch.log1p(-pa - pb)
        return p


    def lnPhi(self, x):
        # computes logarithm of  tail of Z~N(0,1) mitigating numerical roundoff errors
        out = -0.5 * x ** 2 - torch.log(torch.tensor(2, **self.tkwargs)) + torch.log(torch.special.erfcx(x / self.sqrt(2)) + EPS)  # divide by zeros error -> add eps
        return out



def np_lnNormalProb(a, b):
    # computes ln(P(a<Z<b)) where Z~N(0,1) very accurately for any 'a', 'b'
    p = np.zeros_like(a)
    # case b>a>0
    I = a > 0
    if np.any(I):
        pa = np_lnPhi(a[I])
        pb = np_lnPhi(b[I])
        p[I] = pa + np.log1p(-np.exp(pb - pa))
    # case a<b<0
    idx = b < 0
    if np.any(idx):
        pa = np_lnPhi(-a[idx])  # log of lower tail
        pb = np_lnPhi(-b[idx])
        p[idx] = pb + np.log1p(-np.exp(pa - pb))
    # case a < 0 < b
    I = (~I) & (~idx)
    if np.any(I):
        pa = special.erfc(-a[I] / np.sqrt(2)) / 2  # lower tail
        pb = special.erfc(b[I] / np.sqrt(2)) / 2  # upper tail
        p[I] = np.log1p(-pa - pb)
    return p

def np_lnPhi(x):
    # computes logarithm of  tail of Z~N(0,1) mitigating numerical roundoff errors
    out = -0.5 * x ** 2 - np.log(2) + np.log(special.erfcx(x / np.sqrt(2)) + EPS)  # divide by zeros error -> add eps
    return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    d_test = 2
    # random mu and cov
    # mu_test = np.random.rand(d_test)
    # cov_test = 0.5 - np.random.rand(d_test ** 2).reshape((d_test, d_test))
    # cov_test = np.triu(cov_test)
    # cov_test += cov_test.T - np.diag(cov_test.diagonal())
    # cov_test = np.dot(cov_test, cov_test)

    mu_test = torch.zeros(d_test)
    cov_test = torch.eye(d_test)

    # constraints
    lb_test = torch.zeros_like(mu_test) - 1.
    ub_test = torch.ones_like(mu_test) * torch.inf

    # create truncated normal and sample from it
    n_samples_test = 100000
    samples_test = TruncatedMVN(mu_test, cov_test, lb_test, ub_test).sample(n_samples_test)

    idx_test = 1
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    x_test = torch.linspace(-2, 4, 100)
    ax1.plot(x_test, stats.norm.pdf(x_test, mu_test[idx_test], cov_test[idx_test, idx_test]),
             'b--', label='Normal Distribution')
    ax1.set_ylim(bottom=0)
    ax2.hist(samples_test[idx_test, :], 100, color="k", histtype="step",
             label=f'Truncated Normal Distribution, lb={lb_test[0]}, ub={ub_test[0]}')
    ax1.set_xlim([-2, 4])
    ax1.set_yticks([])
    ax2.set_yticks([])
    fig.legend(loc=9, frameon=False)
    plt.show()
    plt.savefig(f'cisbo/test/TMVN.png')
    plt.close()

    info('Done!')