import torch
from cts.vectorized_tmvn.minimax_tilting_tmvn import TruncatedMVN
import time
from dataclasses import dataclass
from torch import Tensor
from torch.distributions import Normal
from logging import debug, warning

PSEUDO_INF = 1e100

def convert_bounds_to_inequality_constraints(bounds):
    r"""Convert bounds into inequality constraints of the form Ax <= b.

    Args:
        bounds: A `2 x d`-dim tensor of bounds

    Returns:
        A two-element tuple containing
            - A: A `2d x d`-dim tensor of coefficients
            - b: A `2d x 1`-dim tensor containing the right hand side
    """
    d = bounds.shape[-1]
    eye = torch.eye(d, dtype=bounds.dtype, device=bounds.device)
    lower, upper = bounds
    lower_finite, upper_finite = bounds.isfinite()
    A = torch.cat([-eye[lower_finite], eye[upper_finite]], dim=0)
    b = torch.cat([-lower[lower_finite], upper[upper_finite]], dim=0).unsqueeze(-1)
    return A, b

def compute_Rmax(
    C: Tensor, # columns are centers of perturbation
    V: Tensor, # columns are perturbation directions
    A: Tensor, # from linear constraints Ax < b
    b: Tensor, # from linear constraints Ax < b
    perturbation_max: Tensor,
) -> Tensor:

    n_points = V.shape[1]
    Alpha = torch.matmul(A, V)    

    B = b.repeat((1, n_points))
    Beta = B - torch.matmul(A, C)

    Alpha_pos_mask = Alpha > 0 # indices where alpha is greater
    ratio = Beta/Alpha
    ratio[~Alpha_pos_mask] = perturbation_max # set non-pos alpha values to large ratio, so they are ignored in min
    #ratio = Alpha_pos_mask*ratio + ~Alpha_pos_mask*(perturbation_max) # set negative alpha values to large ratio, so they are ignored in min
    r_max = torch.min(ratio, dim=0).values
    r_max = r_max.clip(0, perturbation_max)

    return r_max  

def hypercube_get_closest_constraints(A_hat, centered_u, tkwargs):
    """
    Note that all notation in this function follows the notation from the textbook
    Scientific Computing: An Introductory Survey by Heath, Chapter 3, specifically
    section 3.5.

    A_hat is an m x d matrix of constraints
    centered_u is normalized and shifted upper bound for constraints
    d is the dimensionality of the data (number of parameters)
    n is the total number of unique centers we consider
    m is the number of constraints
    tkwargs is a dict that contains data type and device
    """
    m, d = A_hat.shape
    n = centered_u.shape[1]
    # _, closest_boundary_indices = torch.sort(centered_u, dim=0)
    # closest_boundary_indices = closest_boundary_indices.T
    
    l = centered_u[:d]
    u = centered_u[d:]
    l_farther = l > u
    accepted_constraint_mask = torch.cat([~l_farther, l_farther], dim=0)

    accepted_indices = accepted_constraint_mask.nonzero()[:, 0:1].T
    
    A_tilde_mask = torch.zeros((n, m), device=tkwargs['device'], dtype=torch.bool)
    A_tilde_mask = A_tilde_mask.scatter(1, accepted_indices, True)
    A_tilde_mask = A_tilde_mask.unsqueeze(-1).repeat(1, 1, d)

    A_hat = A_hat.repeat(n, 1, 1)
    square_A_tilde = A_hat[A_tilde_mask].reshape(n,d,d)

    for A_tilde_i in square_A_tilde:
        if A_tilde_i.ndim > 2:
            A_tilde_i = A_tilde_i.squeeze(0)
        assert torch.linalg.matrix_rank(A_tilde_i) == A_tilde_i.shape[0]

    return square_A_tilde, A_tilde_mask[:,:,0].squeeze(-1)

# @profile
def householder_get_closest_constraints(A_hat, centered_u, tkwargs):
    """
    Note that all notation in this function follows the notation from the textbook
    Scientific Computing: An Introductory Survey by Heath, Chapter 3, specifically
    section 3.5.

    A_hat is an m x d matrix of constraints
    centered_u is normalized and shifted upper bound for constraints
    d is the dimensionality of the data (number of parameters)
    n is the total number of unique centers we consider
    m is the number of constraints
    tkwargs is a dict that contains data type and device
    """
    m, d = A_hat.shape
    n = centered_u.shape[1]
    sorted_centered_u, closest_boundary_indices = torch.sort(centered_u, dim=0)
    closest_boundary_indices = closest_boundary_indices.T
    epsilon = 1e-4

    I = torch.eye(d, **tkwargs).repeat(n, 1, 1)
    Q_T = I.clone() # initialize Q before any reflections
    U = torch.ones((n,d,d), **tkwargs).triu() # upper triangular matrix filled with ones for convenient construction of Householder vectors
    
    # Permute A_hat to be sorted for each center
    P = torch.zeros(m, m, **tkwargs).repeat(n, 1, 1)
    P = P.scatter(1, closest_boundary_indices.unsqueeze(1), 1)

    A_hat_T = A_hat.repeat(n, 1, 1).transpose(1,2) @ P
    A_hat = A_hat_T.transpose(1,2)

    empty_U_mask = torch.zeros(n, d, device=tkwargs['device'], dtype=torch.bool)
    empty_e_mask = empty_U_mask.clone()

    # A_tilde holds accepted constraints, so we can check if next constraint is in span of already accepted constraints
    A_tilde_mask = torch.zeros((n, m, d), device=tkwargs['device'], dtype=torch.bool)
    A_tilde = A_hat[:, 0, :].reshape(n, 1, d).repeat(1, m, 1)
    
    idx = 0
    n_accept = torch.zeros(n, 1, device=tkwargs['device'], dtype=torch.int64) # number of accepted constraints for each center
    
    while idx < m:
        # 1) check if A_hat[idx] is in span of A_tilde
        a = A_hat[:, idx, :].unsqueeze(-1)
        U_mask = empty_U_mask.scatter(1, n_accept % d, True).unsqueeze(-1).repeat(1, 1, d) # TODO: look into reducing usage of repeat (is this slow?) 
        
        Q2_T_mask = U[U_mask].reshape(n, d, 1).repeat(1, 1, d) # TODO: confirm it is okay to use first row of U_mask for centers with all constraints accepted
        Q2_T = Q_T * Q2_T_mask

        residual_norm = torch.linalg.norm(Q2_T @ a, dim=1) # see Heath textbook section 3.4.5
        in_span = residual_norm < epsilon
        accept = (~in_span) & (n_accept < d) # accept constraints that are not in the span of A_tilde

        # 2) add accepted constraints to A_tilde
        A_tilde_mask[:, idx, :] = accept.repeat(1, d)
        A_tilde[:, idx, :] = A_tilde[:, idx, :] * (~accept).repeat(1, d) + a.squeeze() * accept.repeat(1, d) # update only accepted constraints

        # 3) update Q_T
        a2_mask = U[U_mask].reshape(n, d, 1)
        e_mask = empty_e_mask.scatter(1, n_accept % d, True).unsqueeze(-1).repeat(1, 1, d) # TODO: see if this modulus solved the need for extra rows

        a = Q_T @ a
        a2 = a * a2_mask  # top n_accept elements are set to 0
        ak = a[e_mask[:,:d,0].unsqueeze(-1)]
        sign = -torch.sign(ak + 1e-6) # add tiny positive number to get +1 sign for ak = 0
        alpha = torch.linalg.norm(a2, dim=1).unsqueeze(-1) * sign.reshape(n, 1, 1)
        e = I[e_mask].reshape(n, d, 1)

        v = a2 - alpha*e
        vT = v.transpose(1,2)
        vTv = vT @ v

        not_accept_mask = torch.isclose(vTv, torch.tensor(0, **tkwargs)).squeeze() | (~accept).squeeze()
        # set these H_i to be identity
        v[not_accept_mask] = 0
        vTv[not_accept_mask] = 1

        Q_T = Q_T - 2/vTv * v @ vT @ Q_T # avoid explicity formation of H

        n_accept += accept
        idx += 1

        # decide if we should continue
        if (n_accept >= d).all():
            break

    square_A_tilde = A_hat[A_tilde_mask].reshape(n, d, d)

    # for A_tilde_i in square_A_tilde:
    #     A_tilde_i = A_tilde_i.squeeze()
    #     assert torch.linalg.matrix_rank(A_tilde_i) == A_tilde_i.shape[0]
    
    accepted_constraint_mask = A_tilde_mask[:,:,0].squeeze(-1)
    accepted_centered_u = sorted_centered_u.T[accepted_constraint_mask].reshape(n, d)
    # return square_A_tilde, A_tilde_mask[:,:,0].squeeze(-1)
    return square_A_tilde, accepted_centered_u


@dataclass
class TMVN_Cache:
    key: torch.Tensor # rows are centers 
    scaled_L: torch.Tensor
    lb: torch.Tensor
    ub: torch.Tensor
    psistar: torch.Tensor
    mu: torch.Tensor # from psistar optimization
    # A_hat Lp and P used in TMVN_CylindricalSampler, not by dist sampler
    A_hat: torch.Tensor
    Lp: torch.Tensor
    P: torch.Tensor
    centered_u: torch.Tensor


class TMVN_CylindricalSampler:
    def __init__(self, d, device, sigma_init=1.0) -> None:

        self.tkwargs = {
            "dtype": torch.double,
            "device": device,
        }

        self.d = d
        self.reset_cache()
        self.sigma = torch.tensor(sigma_init, **self.tkwargs)
        
        self.max_cache_size = 10
        self.R = None

    def reset_cache(self):
        d = self.d
        # populate cache with empty tensors
        key = torch.empty((0, d), **self.tkwargs)
        scaled_L = torch.empty((0, d, d), **self.tkwargs)
        lb = torch.empty((0, d, 1), **self.tkwargs)
        ub = torch.empty((0, d, 1), **self.tkwargs)
        psistar = torch.empty(0, **self.tkwargs)
        mu = torch.empty((0, d-1), **self.tkwargs)
        A_hat = torch.empty((0, d, d), **self.tkwargs)
        Lp = torch.empty((0, d, d), **self.tkwargs)
        P = torch.empty((0, d, d), **self.tkwargs)
        centered_u = torch.empty((0, d), **self.tkwargs)
        
        self.cache = TMVN_Cache(key, scaled_L, lb, ub, psistar, mu, A_hat, Lp, P, centered_u)

    
    def get_cache_hits(self, centers):

        cache = self.cache
        # use_mm_for_euclid_dist=True for torch.cdist
        if len(cache.key) > 0:
            K = self.cache.key
            C = centers

            # distances = torch.cdist(K, C, compute_mode='use_mm_for_euclid_dist')
            distances = torch.cdist(C, K, compute_mode='use_mm_for_euclid_dist')

            hit_mask = torch.isclose(distances, torch.tensor(0).to(distances), atol=1e-6)
    
            # row indicates center index, col indicates key index
            # assert (hit_mask.sum(dim=0) <= 1).all() # maximum of one hit per center
            if not (hit_mask.sum(dim=1) <= 1).all():
                warning("Multiple hits in cache, this could be caused by evaluating multiple configs with very similar param values.")
                warning("Arbitrarily selecting first hit in cache...")
                # arbitrarily select first cache hit
                sparse_hit_mask = torch.zeros_like(hit_mask, dtype=torch.bool)
                for i in range(len(C)):
                    for j in range(len(K)):
                        if hit_mask[i, j]: # hit for center i and key j
                            sparse_hit_mask[i, j] = True
                            break # move on to next center, only want one hit
                hit_mask = sparse_hit_mask
            
            assert (hit_mask.sum(dim=1) <= 1).all() # maximum of one hit per center

            centers_hit_mask = hit_mask.any(dim=1) # used to remove hits from computation
            hit_indices = hit_mask.nonzero()[:,1]

            if centers_hit_mask.any():
                debug('cache hit')
            else:
                # no hits
                if self.max_cache_size == 1:
                    self.reset_cache()
            
        else:
            centers_hit_mask = torch.zeros(centers.shape[0], device=centers.device).to(torch.bool)
            hit_indices = torch.tensor([],dtype=torch.int64)

        # update cache keys
        non_hit_centers = centers[~centers_hit_mask] 
        # cache.key = torch.cat([cache.key, non_hit_centers], dim=0)
        if (~centers_hit_mask).any():
            self.cache.key = torch.cat([self.cache.key, non_hit_centers], dim=0)
            
        return centers_hit_mask, hit_indices


    def get_constraints(self, A, b, c):
        
        if len(c) > 0:
            start_t = time.perf_counter()
            new_A_hat, new_centered_u = self.vectorized_get_centered_constraints(c.T, A, b)
            end_t = time.perf_counter()
            debug(f'Constraint selection time: {end_t-start_t}')
        else:
            new_A_hat = None
            new_centered_u = None

        return new_A_hat, new_centered_u

    def add_cached_constraints(self, new_A_hat, new_centered_u, hit_mask, hit_indices):
        
        cache = self.cache
        d = self.d
        n = hit_mask.shape[0]
        A_hat = torch.zeros((n, d, d), **self.tkwargs)
        centered_u = torch.zeros((n, d), **self.tkwargs)

        A_hat[hit_mask] = cache.A_hat[hit_indices]
        centered_u[hit_mask] = cache.centered_u[hit_indices]

        if (~hit_mask).any():
            A_hat[~hit_mask] = new_A_hat
            centered_u[~hit_mask] = new_centered_u

            # update cache
            cache.A_hat = torch.cat([cache.A_hat, new_A_hat], dim=0)
            cache.centered_u = torch.cat([cache.centered_u, new_centered_u], dim=0)

        return A_hat, centered_u

    def get_factors(self, A_hat, centered_u):
        """
        Get Cholesky factor, permutation matrix, and tilting params
        """

        # only compute factors for non hit centers
        if A_hat is not None:
                
            d = self.d

            cov = A_hat @ A_hat.transpose(1,2)
            # zero_mu = torch.zeros_like(non_hit_centers, **self.tkwargs) # constraints are centered -> assume 0 mu
            zero_mu = torch.zeros(A_hat.shape[0], d, **self.tkwargs) # constraints are centered -> assume 0 mu
            tmvn_dist = TruncatedMVN(zero_mu, cov, -PSEUDO_INF*torch.ones_like(centered_u), centered_u)

            opt_t, decomp_t = tmvn_dist.vectorized_compute_factors()
            debug(f'time opt: {opt_t}') # TODO logger debug
            debug(f'time decomp: {decomp_t}')

            # collect data for cache
            psistar = tmvn_dist.psistar
            mu = tmvn_dist.mu
            scaled_L = tmvn_dist.L
            Lp = tmvn_dist.unscaled_L
            ub = tmvn_dist.ub
            P = tmvn_dist.perm
        
            # D = torch.diagonal(dist.unscaled_L, dim1=-2, dim2=-1)
            # assert torch.isclose(new_ub*D.unsqueeze(-1), new_P @ new_centered_u.unsqueeze(-1)).all()
        else:
            Lp, ub, scaled_L, P, mu, psistar = None, None, None, None, None, None

        return Lp, ub, scaled_L, P, mu, psistar

    def add_cached_factors(self, n, new_Lp, new_ub, new_scaled_L, new_P, new_mu, new_psistar, hit_mask, hit_indices):
        
        d = self.d
        
        # Form data tensors for sampler. Start filled with zeros and then populate.
        scaled_L = torch.zeros((n, d, d), **self.tkwargs)
        ub = torch.zeros((n, d, 1), **self.tkwargs)
        psistar = torch.zeros(n, **self.tkwargs)
        mu = torch.zeros((n, d-1), **self.tkwargs)
        Lp = torch.zeros((n, d, d), **self.tkwargs)
        P = torch.zeros((n, d, d), **self.tkwargs)

        # combine cache hits and new factors
        cache = self.cache
        
        # fill in cache hits
        scaled_L[hit_mask] = cache.scaled_L[hit_indices]
        ub[hit_mask] = cache.ub[hit_indices]
        psistar[hit_mask] = cache.psistar[hit_indices]
        mu[hit_mask] = cache.mu[hit_indices]
        P[hit_mask] = cache.P[hit_indices]
        Lp[hit_mask] = cache.Lp[hit_indices]

        if (~hit_mask).any():
            # fill in new data
            scaled_L[~hit_mask] = new_scaled_L
            ub[~hit_mask] = new_ub
            psistar[~hit_mask] = new_psistar
            mu[~hit_mask] = new_mu
            P[~hit_mask] = new_P
            Lp[~hit_mask] = new_Lp

            # update cache
            cache.scaled_L = torch.cat([cache.scaled_L, new_scaled_L], dim=0)
            cache.ub = torch.cat([cache.ub, new_ub], dim=0)
            cache.psistar = torch.cat([cache.psistar, new_psistar], dim=0)
            cache.mu = torch.cat([cache.mu, new_mu], dim=0)
            cache.Lp = torch.cat([cache.Lp, new_Lp], dim=0)
            cache.P = torch.cat([cache.P, new_P], dim=0)
            
        return Lp, ub, scaled_L, P, mu, psistar
    
    def get_tmvn_sampler(self, n, mu, scaled_L, ub, psistar):
        zero_mu = torch.zeros((1, self.d), **self.tkwargs) # dummy to pass tkwargs
        tmvn_dist = TruncatedMVN(zero_mu, None, None, None)
        # set the dist data
        tmvn_dist.n = n 
        tmvn_dist.mu = mu
        tmvn_dist.L = scaled_L
        tmvn_dist.lb = -PSEUDO_INF * torch.ones_like(ub) # we assume all constraints are upper bounds
        tmvn_dist.ub = ub
        tmvn_dist.psistar = psistar
        
        return tmvn_dist
        

    def sample_trunc_normal(self, n_discrete_points, centered_l, centered_u):
        u = torch.rand(
            (n_discrete_points, 1), **self.tkwargs
        )
        
        # compute z-score of bounds
        alpha = centered_l
        beta = centered_u
        normal = Normal(0, 1)
        cdf_alpha = normal.cdf(alpha)
        perturbation = normal.icdf(cdf_alpha + u * (normal.cdf(beta) - cdf_alpha))

        return perturbation

    def one_dimensional_tmvn_dir(self, unique_centers, counts, A_hat, centered_l, centered_u):
        dirs = []
        for i in len(unique_centers):
            samples = self.sample_trunc_normal(counts[i], centered_l[i], centered_u[i])
            dirs.append(samples/torch.linalg.norm(samples, dim=1).unsqueeze(-1))

        return torch.cat(dirs, dim=0).T

    
    def vectorized_get_centered_constraints(self, c, A, b):
        """
        c, A, l, u tensors in matrix or col vec form

        l < A (c + z) < u 
        transform to
        l_hat < A_hat (c + z) < u_hat
        
        d-dimensional space
        n center points
        m constraints
        
        """
        
        d = c.shape[0]
        n = c.shape[1]
        tkwargs = {
            "dtype": c.dtype,
            "device": c.device
        }
        
        a_norms = torch.linalg.norm(A, dim=1) .unsqueeze(1)
        A_hat = A / a_norms # rows have unit length, in direction of constraint boundary normal vector
        b_hat = b / a_norms
        
        # get distance from c to boundaries
        centered_u = b_hat - A_hat @ c

        assert (centered_u >= 0.0).all() # confirm centers are valid

        # exclude constraints that are outside of trust region radius
        constraints_outside_TR = centered_u > self.R
        centered_u[constraints_outside_TR] = 1e100

        # untruncate_threshold = 0.25
        # constraints_untruncated = centered_u > untruncate_threshold
        # centered_u[constraints_untruncated] = 1e100

        # A_tilde, accepted_constraint_mask = vectorized_QR_get_closest_constraints(A_hat, centered_u, d, n, m, tkwargs)
        if A.shape[0] == A.shape[1]:
            A_tilde = A.repeat(n, 1, 1)
            accepted_constraint_mask = torch.ones_like(centered_u.T, dtype=torch.bool)
            centered_u = centered_u.T[accepted_constraint_mask].reshape(n, d)
        else:
            # check if A is of the form [-I, I]^T, which is the case for hypercube constraints that consider only upper bounds
            if n == 1 and torch.isclose(A[:d], -torch.eye(d, **tkwargs)).all() and torch.isclose(A[d:], torch.eye(d, **tkwargs)).all():
                A_tilde, accepted_constraint_mask = hypercube_get_closest_constraints(A_hat, centered_u, tkwargs)
                centered_u = centered_u.T[accepted_constraint_mask].reshape(n, d)
            else:
                # A_tilde, accepted_constraint_mask = householder_get_closest_constraints(A_hat, centered_u, tkwargs)
                A_tilde, centered_u = householder_get_closest_constraints(A_hat, centered_u, tkwargs)

        centered_u /= self.sigma

        return A_tilde, centered_u
    
    def transform_warped_samples_multiple_centers(self, xp, A_hat, Lp, P, centered_u, counts, parallel = False):
        """
        Used in MORBO whenever we need to perform CTS with multiple centers of peturbation.
        Calling this function with parallel=True can cause memory issues on certain systems.        
        """
        if parallel:
            A_hat = A_hat.repeat_interleave(counts, dim=0)
            Lp = Lp.repeat_interleave(counts, dim=0)
            P = P.repeat_interleave(counts, dim=0)
            I = torch.eye(self.d, **self.tkwargs).unsqueeze(0).repeat(counts.sum(),1,1)
            centered_u = centered_u.repeat_interleave(counts, dim=0)
            
            torch.cuda.empty_cache()
            
            # assert torch.isclose(Lp.tril(), Lp).all()
            Lp_inv = torch.linalg.solve_triangular(Lp, I, upper=False)
            Qp_T = Lp_inv @ P @ A_hat # Qp.T = Lp^{-1} @ PA

            # confirm that Qp is indeed orthogonal
            # assert torch.isclose(Qp_T @ Qp_T.transpose(1,2), I).all()
            # assert torch.isclose(Lp @ Qp_T, P @ A_hat).all()
            Qp = Qp_T.transpose(1,2)

            z = Qp @ xp

            assert (A_hat.unsqueeze(0) @ z < centered_u.unsqueeze(-1)).all()

        else:
            start = 0
            z_serial = torch.empty((0, self.d, 1), **self.tkwargs)
            I = torch.eye(self.d, **self.tkwargs)
            for i, stop in enumerate(counts.cumsum(0)):
                xp_i = xp[start:stop]
                A_hat_i = A_hat[i]
                Lp_i = Lp[i]
                P_i = P[i]
                centered_u_i = centered_u[i]
                assert len(xp_i) == counts[i]

                start = stop # update starting index

                Lp_i = Lp_i.squeeze()
                Lp_inv = torch.linalg.solve_triangular(Lp_i, I, upper=False)

                A_hat_i = A_hat_i.squeeze()
                P_i = P_i.squeeze()
                PA = P_i @ A_hat_i

                Qp_T = Lp_inv @ PA # Qp.T = Lp^{-1} @ PA
                Qp = Qp_T.T

                z_i = Qp.unsqueeze(0) @ xp_i

                assert (A_hat_i.unsqueeze(0) @ z_i < centered_u_i.unsqueeze(-1)).all()
                z_serial = torch.cat([z_serial, z_i], dim=0)
            
            z = z_serial

            assert (A_hat.repeat_interleave(counts, dim=0) @ z < centered_u.repeat_interleave(counts, dim=0).unsqueeze(-1)).all()
        
        z = z * self.sigma
        
        return z
    
    
    def transform_warped_samples_single_center(self, xp, A_hat, Lp, P, centered_u):
        # when using hypercube constraints, A_hat = I
        # need to repeat each relevant matrix based on counts (use repeat_interleave)
        Lp = Lp.squeeze()
        I = torch.eye(self.d, **self.tkwargs)
        # want z (normally distributed with mu = 0, cov = I, such that l < Az < u)
        # We have the following relationships which we use to solve for z
        # xp = Qp.T @ z
        # Lp @ Qp.T = PA permuted A matrix
        
        # assert torch.isclose(Lp.tril(), Lp).all()
        Lp_inv = torch.linalg.solve_triangular(Lp, I, upper=False)
        # del Lp
        # del I

        A_hat = A_hat.squeeze()
        P = P.squeeze()
        PA = P @ A_hat
        # del P
        # del A_hat

        # Qp_T = Lp_inv @ P @ A_hat # Qp.T = Lp^{-1} @ PA
        Qp_T = Lp_inv @ PA # Qp.T = Lp^{-1} @ PA
        # del PA

        # confirm that Qp is indeed orthogonal
        # assert torch.isclose(Qp_T @ Qp_T.transpose(1,2), I).all()
        # assert torch.isclose(Lp @ Qp_T, P @ A_hat).all()
        Qp = Qp_T.T
        # del Qp_T

        # A_inv = Qp @ Lp_inv @ P
        # D = torch.diagonal(Lp, dim1=-2, dim2=-1) # batch diagonal
        # x = (P.T).unsqueeze(0) @ (D.unsqueeze(-1) * xp) # un-permute and rescale xp
        # z = A_inv.unsqueeze(0) @ x
        z = Qp.unsqueeze(0) @ xp
        # del Qp

        assert (A_hat.unsqueeze(0) @ z < centered_u.unsqueeze(-1)).all()
        z = z * self.sigma
        
        return z
    
    def transform_warped_samples(self, xp, A_hat, Lp, P, centered_u, counts):
        
        if len(counts) > 1:
            return self.transform_warped_samples_multiple_centers(xp, A_hat, Lp, P, centered_u, counts)
        else:
            return self.transform_warped_samples_single_center(xp, A_hat, Lp, P, centered_u)
    
    
    def validate_invariants(self, counts, Lp, P, xp, ub, centered_u):
        
        D = torch.diagonal(Lp, dim1=-2, dim2=-1) # batch diagonal
        if len(counts) > 1:
            D = D.repeat_interleave(counts, dim=0)
            Lp = Lp.clone().repeat_interleave(counts, dim=0)
            P = P.clone().repeat_interleave(counts, dim=0)
            ub = ub.clone().repeat_interleave(counts, dim=0)
            centered_u = centered_u.clone().unsqueeze(-1).repeat_interleave(counts, dim=0)
            
            assert (Lp @ xp < P @ centered_u).all()
            assert torch.isclose(ub, (P @ centered_u)/D.unsqueeze(-1)).all()
        else:
            assert (Lp @ xp < P @ centered_u.clone().unsqueeze(-1)).all()
            assert torch.isclose(ub, (P @ centered_u.clone().unsqueeze(-1))/D.unsqueeze(-1)).all()
    
    def sample_dir_tmvn_with_radius(self, unique_centers, counts, A, b, R):
        """
        
        A is (n x d) with n constraints
        b is an upper bound vector of shape (n x 1) such that the valid search space is {x s.t. Ax < b}
        
        """
        if len(unique_centers.shape) == 1:
            unique_centers = unique_centers.unsqueeze(0)
        centers = unique_centers
        d = centers.shape[1]
        n = centers.shape[0]

        # check cache for hits        
        hit_mask, hit_indices = self.get_cache_hits(centers)

        # gather centers for which we need to compute tilting params
        new_centers = centers[~hit_mask] 
        
        # new_A_hat, new_centered_u = self.get_constraints(A, b, new_centers, hit_mask, hit_indices)
        new_A_hat, new_centered_u = self.get_constraints(A, b, new_centers)
        A_hat, centered_u = self.add_cached_constraints(new_A_hat, new_centered_u, hit_mask, hit_indices)
        
        # check if we are in the simpler setting of 1-dimensional truncated normal
        if d == 1: # sometimes the case with BAxUS
            return self.one_dimensional_tmvn_dir(unique_centers, counts, A_hat, -PSEUDO_INF * torch.ones_like(centered_u), centered_u)
        
        Lp, ub, scaled_L, P, mu, psistar = self.get_factors(new_A_hat, new_centered_u)
        Lp, ub, scaled_L, P, mu, psistar = self.add_cached_factors(n, Lp, ub, scaled_L, P, mu, psistar, hit_mask, hit_indices) 
        
        # assert torch.isclose(ub, P @ centered_u.unsqueeze(-1)).all()
        
        tmvn_sampler = self.get_tmvn_sampler(n, mu, scaled_L, ub, psistar)

        # sample
        start_t = time.perf_counter()
        xp = tmvn_sampler.vectorized_sample(counts) # xp are warped and permuted for efficient sampling - these transformations must be corrected
        end_t = time.perf_counter()

        rejection_sampling_time = end_t - start_t
        debug(f'{rejection_sampling_time=}')
        
        self.validate_invariants(counts, Lp, P, xp, ub, centered_u)
        
        z = self.transform_warped_samples(xp, A_hat, Lp, P, centered_u, counts)
        pert_directions = z/torch.linalg.norm(z, dim=1).unsqueeze(1)

        return pert_directions.squeeze().T

    def forward_STuRBO(
        self,
        best_X: Tensor,
        A: Tensor,
        b: Tensor,
        Rmax: Tensor,
        sigma: float,
        n_discrete_points: int,
    ) -> Tensor:
    
        # NOTE: This is all happening in the normalized global coordinates
        if best_X.shape[0] == 1:
            X_cand = best_X.repeat(n_discrete_points, 1)
            unique_centers = X_cand[0].unsqueeze(0)
            counts = torch.tensor([n_discrete_points], device=X_cand.device)
        else: 
            rand_indices = torch.randint(
                best_X.shape[0], (n_discrete_points,), device=best_X.device
            )
            # we count the unique centers of perturbation to prevent redundant factorizations & optimizations in
            # truncated mvn sampler
            sorted_indices, _ = rand_indices.sort()
            unique_indices, counts = torch.unique(sorted_indices, return_counts=True)
            X_cand = best_X[sorted_indices]
            unique_centers = best_X[unique_indices]

        self.R = Rmax
        
        if not self.sigma == sigma:
            self.sigma = sigma
            self.reset_cache()
        
        V = self.sample_dir_tmvn_with_radius(unique_centers, counts, A, b, Rmax)

        r_max = compute_Rmax(X_cand.T, V, A, b, perturbation_max=Rmax)
        r = torch.rand_like(r_max) * r_max
        R = torch.diag(r)

        mean_sample_dist = r.mean()
        debug(f'{mean_sample_dist=}')

        X_cand = X_cand + torch.matmul(V, R).t()

        assert (torch.matmul(A, X_cand.T) <= b + 1e-6).all()
        return X_cand
