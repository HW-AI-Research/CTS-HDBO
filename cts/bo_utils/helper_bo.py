"""
Functions used in bayesian optimization of qNEHVI
"""
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from datetime import datetime
import os
import torch
import yaml
import pickle
from botorch.test_functions.base import ConstrainedBaseTestProblem
from cts.morbo.benchmark_function import BenchmarkFunction
from logging import info

def save_intermediate(train_x, train_obj, run_dir):
    '''
    This function saves parameters (input) and metrics (output) in the run_dir directory.
    Call this function in the middle of training.
    '''
    torch.save(train_x, f'{run_dir}/results/intermediate/train_x.pt')
    torch.save(train_obj, f'{run_dir}/results/intermediate/train_obj.pt')


def save_indicator(indicator, run_dir):
    '''
    This function saves the hyper volume in the run_dir directory
    '''
    hvs_path = f'{run_dir}/results/indicator.p'
    with open(hvs_path, "wb") as file:
        pickle.dump(indicator, file)

def setup_run_dir(experiment_config, problem):
    """
    Set up run dir and save initial logging info.
    """
    EXP_NAME = experiment_config['EXP_NAME']
    
    timestamp = datetime.now()
    experiment_config['timestamp'] = timestamp
    
    base_log_dir = f'{EXP_NAME}'
    info(f'===> Logging to {base_log_dir}')
    os.makedirs(f'{EXP_NAME}/results', exist_ok=True)
    os.makedirs(f'{EXP_NAME}/results/intermediate', exist_ok=True)
    os.makedirs(f'{EXP_NAME}/results/pareto', exist_ok=True)
    os.makedirs(f'{EXP_NAME}/configs', exist_ok=True)
    
    config_save_path = f'{EXP_NAME}/configs/experiment_config.yaml'
    
    with open(config_save_path, 'w+') as f:
        yaml.dump(experiment_config, f, allow_unicode=True, sort_keys=False)
    

    return f'{EXP_NAME}'

def compute_indicator(problem, train_obj, num_objectives):
    if isinstance(problem, ConstrainedBaseTestProblem) and problem.objective is not None:
        feasible_obj = problem.objective(train_obj)
        non_nan_mask = ~torch.isnan(feasible_obj).any(dim=-1)
        feasible_obj = feasible_obj[non_nan_mask]
    else:
        feasible_obj = train_obj
        
    if num_objectives > 1:
        # compute hypervolume
        bd = DominatedPartitioning(ref_point=problem.ref_point, Y=feasible_obj)
        volume = bd.compute_hypervolume().item()
        indicator = volume
    else: # single metrics
        indicator = torch.max(feasible_obj)
    
    return indicator

def log_progress(problem, num_objectives, iteration, elapsed_time, indicators, train_obj, alg_name):
    '''
    This function updates the hyper volume list for the BO algorithm
    '''
    indicator = compute_indicator(problem, train_obj, num_objectives)
    indicators.append(indicator)
    indicator_type = 'Hypervolume' if num_objectives > 1 else 'f_value'
    
    info(f" {alg_name} --- Batch {iteration:>2}: {indicator_type} = {indicators[-1]:>4.2f}; time = {elapsed_time:>4.2f}.")

    return indicators