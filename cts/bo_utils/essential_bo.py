from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.transforms import unnormalize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_fully_bayesian_model_nuts
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.optimize import optimize_acqf
import torch
import time
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from cts.morbo.benchmark_function import BenchmarkFunction
from logging import debug

def fit_SingleTask_model(train_x, train_obj, bounds, model_cfg):
    
    # normalize input and standarize output
    in_tf = Normalize(d=train_x.shape[-1], bounds=bounds)
    out_tf = Standardize(m=train_obj.shape[-1])

    debug("fitting GP...")
    model = SingleTaskGP(train_x, train_obj, input_transform=in_tf, outcome_transform=out_tf)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    debug("done fitting")
    
    return model

def fit_SAAS_GP_model(train_x, train_obj, bounds, model_cfg):
    # define models for objective and constraint
    # ****** TODO: get this parameters from yaml
    if 'MCMC_WARMUP_STEPS' in model_cfg:
        WARMUP_STEPS = model_cfg['MCMC_WARMUP_STEPS']
        NUM_SAMPLES = model_cfg['MCMC_NUM_SAMPLES']
        THINNING = model_cfg['MCMC_THINNING']
    else:
        WARMUP_STEPS = 128 #512
        NUM_SAMPLES = 128 #256
        THINNING = 16
    
    debug('Starting SAAS GP fitting...')
    
    # normalize input and standarize output
    in_tf = Normalize(d=train_x.shape[-1], bounds=bounds)
    out_tf = Standardize(m=1)

    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]
        train_yvar = torch.full_like(train_y, 1e-6)
        model_saasbo = SaasFullyBayesianSingleTaskGP(train_x, train_y, train_yvar, input_transform=in_tf, outcome_transform=out_tf)
        
        fit_fully_bayesian_model_nuts(
            model_saasbo, warmup_steps=WARMUP_STEPS, num_samples=NUM_SAMPLES, thinning=THINNING, disable_progbar=True)
        
        models.append(model_saasbo)

    model = ModelListGP(*models)
    debug('SAAS GP fitting complete.')
    
    return model

def optimize_qnehvi_and_get_observation(model, train_x, problem, bounds, acq_cfg):
    debug('acquisition...')
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point 
        X_baseline=train_x,
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
    )
    
    return optimize_acq_fn_and_get_observation(acq_func, problem, bounds, acq_cfg)

def optimize_qnei_and_get_observation(model, train_x, problem, bounds, acq_cfg):
    debug('acquisition...')
    acq_func = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        prune_baseline=True,  # prune points that are highly unlikely to be optimal
    )

    return optimize_acq_fn_and_get_observation(acq_func, problem, bounds, acq_cfg)

def optimize_acq_fn_and_get_observation(acq_func, problem, bounds, acq_cfg):
    
    # optimize
    new_x, acq_values = optimize_acqf(
        acq_func,
        bounds=bounds,
        q=acq_cfg['BATCH_SIZE'],
        num_restarts=acq_cfg['NUM_RESTARTS'],
        raw_samples=acq_cfg['RAW_SAMPLES'],
        options={"batch_limit": 1, "maxiter": 200},
        sequential=True,
        inequality_constraints=None,
        )

    # observe new values 
    if isinstance(problem, BenchmarkFunction):
        new_obj = problem(new_x)
        new_obj = new_obj.cpu().numpy()
    else:    
        new_obj = problem(new_x.cpu().numpy())
        
    debug('acquisition done')

    return new_x, new_obj

class SobolSampler:
    def __init__(self, dim, bounds) -> None:
        self.SOBOL_ENG = SobolEngine(dim, scramble=True)
        self.bounds = bounds
        self.tkwargs = {
            "dtype": bounds.dtype,
            "device": bounds.device
        }

    def sample(self, n):

        candidates = self.SOBOL_ENG.draw(n).to(**self.tkwargs)
        new_x =  unnormalize(candidates.detach(), bounds=self.bounds)
        
        return new_x
    
    def sample_and_eval(self, n, problem):
        
        new_x =  self.sample(n)

        if isinstance(problem, BenchmarkFunction):
            new_obj = problem(new_x)
            new_obj = new_obj.cpu().numpy()
        else:    
            new_obj = problem(new_x.cpu().numpy())
        return new_x, new_obj
    
ACQ_FNS = {
    'qnehvi': optimize_qnehvi_and_get_observation,
    'qnei': optimize_qnei_and_get_observation,
}

MODEL_FIT_FNS = {
    'SingleTaskGP': fit_SingleTask_model,
    'SAAS': fit_SAAS_GP_model,
}

