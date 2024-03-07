from botorch.utils.sampling import HitAndRunPolytopeSampler
import torch
import time
from cts.cylindrical_ts.sampling_utils import TMVN_CylindricalSampler
import math

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

def sample_polytope(A, b, n_samples):
    sampler = HitAndRunPolytopeSampler(inequality_constraints = (A, b))
    samples = sampler.draw(n=n_samples)
    
    return samples

def generate_random_constraints(d):
    """
    Generate random linearly independent linear constraints in d-dimensional space
    """
    A = torch.zeros(d, d, device='cuda:0', dtype=torch.double)
    b = torch.zeros(d, 1, device='cuda:0', dtype=torch.double)
    
    for i in range(d):
        if i == 0:
            A[i] = torch.rand_like(A[i])
            b[i] = torch.rand(1)
        else:
            A[i] = A[i-1] # set A[i] so that while loop will be triggered
            while torch.linalg.matrix_rank(A[:i+1].T) != i+1:
                A[i] = torch.rand_like(A[i]) - 0.5
                
            b[i] = torch.rand(1) - 0.5
            
    return A, b

if __name__ == '__main__':
    
    d = 80
    for i in range(100):
        A, b = generate_random_constraints(d)
        c = sample_polytope(A, b, 5)
        bounds = torch.cat([c.min(dim=0).values.unsqueeze(0) - 1e-3, c.max(dim=0).values.unsqueeze(0) + 1e-3])
        CTS_sampler = sampler = TMVN_CylindricalSampler(d=d, device=A.device, sigma_init=0.125)
        
        A2, b2 = convert_bounds_to_inequality_constraints(bounds=bounds)
        A = torch.cat([A, A2], dim=0)
        b = torch.cat([b, b2], dim=0)
            
        x = CTS_sampler.forward_STuRBO(best_X=c, A=A, b=b, Rmax=math.sqrt(d), sigma=0.125, n_discrete_points=500)
    
    