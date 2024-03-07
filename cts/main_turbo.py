import torch
import os
import yaml
import numpy as np
from cts.turbo import TurboM, Turbo1
from cts.turbo import STurboM, STurbo1
import pickle
from cts.utils.parsing import common_arg_parser
from cts.bo_utils.helper_bo import (
    setup_run_dir, 
)
from cts.baxus.util.parsing import fun_mapper, benchmark_loader
from logging import info

BASE_SEED = 12346

def main(args):

    assert args.algorithm in ['turbo', 'cts-turbo', 'cts-bo', 'bock', 'thompson_sampling']
    alg = args.algorithm

    benchmark_loader(args.function, args=None)
    problem = fun_mapper()[args.function](dim=args.input_dim, noise_std=0.0)

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu"),
    }

    with open(f'{args.exp_config}', 'r') as fp:
        experiment_config = yaml.safe_load(fp)
           
    BASE_EXP_NAME = args.logdir

    EVAL_BUDGET = args.max_evals
    RANDOM_INIT_SAMPLES = args.n_init
    BATCH_SIZE = experiment_config['BATCH_SIZE']

    lb = problem.lb_vec
    ub = problem.ub_vec

    trials = args.num_repetitions

    for trial in range(trials):
        info(f'============= Trial {trial} =============')
        if trials > 1:
            EXP_NAME = BASE_EXP_NAME + f'/trial_{trial}'
            os.makedirs(f'{EXP_NAME}')
        else:
            EXP_NAME = BASE_EXP_NAME
            
        experiment_config['EXP_NAME'] = EXP_NAME
        RUN_DIR = setup_run_dir(experiment_config, problem)
        # call helper functions to generate initial training data and initialize model
        n_TR = args.n_trust_regions
        if n_TR > 1:
            if alg == 'turbo' or alg == 'bock' or alg == 'thompson_sampling':
                turbo = TurboM(
                    f=problem,  # Handle to objective function
                    lb=lb,  # Numpy array specifying lower bounds
                    ub=ub,  # Numpy array specifying upper bounds
                    n_init=RANDOM_INIT_SAMPLES//n_TR,  # Number of initial bounds from an Symmetric Latin hypercube design
                    max_evals=EVAL_BUDGET,  # Maximum number of evaluations
                    n_trust_regions=n_TR,  # Number of trust regions
                    batch_size=experiment_config['BATCH_SIZE'],  # How large batch size TuRBO uses
                    verbose=experiment_config['verbose'],  # Print information from each batch
                    failtol=experiment_config['failure_streak'] if 'failure_streak' in experiment_config else None,
                    length_init=experiment_config['TR_INIT_L'] if 'TR_INIT_L' in experiment_config else 0.8,
                    scale_TR_sides=experiment_config['scale_TR_sides'] if 'scale_TR_sides' in experiment_config else True,
                    RAASP=experiment_config['RAASP'] if 'RAASP' in experiment_config else True, # random axis-aligned subspace perturbations
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    min_cuda=0,  # Run on the CPU for small datasets
                    device=tkwargs['device'],  # "cpu" or "cuda"
                    dtype="float64",  # float64 or float32
                )
            elif alg == 'cts-turbo' or alg == 'cts-bo':
                turbo = STurboM(
                    f=problem,  # Handle to objective function
                    lb=lb,  # Numpy array specifying lower bounds
                    ub=ub,  # Numpy array specifying upper bounds
                    n_init=RANDOM_INIT_SAMPLES//n_TR,  # Number of initial bounds from an Symmetric Latin hypercube design
                    max_evals=EVAL_BUDGET,  # Maximum number of evaluations
                    n_trust_regions=n_TR,  # Number of trust regions
                    batch_size=experiment_config['BATCH_SIZE'],  # How large batch size TuRBO uses
                    verbose=experiment_config['verbose'],  # Print information from each batch
                    failtol=experiment_config['failure_streak'] if 'failure_streak' in experiment_config else None,
                    length_init=experiment_config['TR_INIT_L'] if 'TR_INIT_L' in experiment_config else 0.8,
                    scale_TR_sides=experiment_config['scale_TR_sides'] if 'scale_TR_sides' in experiment_config else True,
                    RAASP=experiment_config['RAASP'] if 'RAASP' in experiment_config else True, # random axis-aligned subspace perturbations
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    min_cuda=0,  # Run on the CPU for small datasets
                    device=tkwargs['device'],  # "cpu" or "cuda"
                    dtype="float64",  # float64 or float32
                )
        else:
            if alg == 'turbo' or alg == 'bock' or alg == 'thompson_sampling':
                turbo = Turbo1(
                    f=problem,  # Handle to objective function
                    lb=lb,  # Numpy array specifying lower bounds
                    ub=ub,  # Numpy array specifying upper bounds
                    n_init=RANDOM_INIT_SAMPLES,  # Number of initial bounds from an Latin hypercube design
                    max_evals=EVAL_BUDGET,  # Maximum number of evaluations
                    batch_size=experiment_config['BATCH_SIZE'],  # How large batch size TuRBO uses
                    verbose=experiment_config['verbose'],  # Print information from each batch
                    failtol=experiment_config['failure_streak'] if 'failure_streak' in experiment_config else None,
                    length_init=experiment_config['TR_INIT_L'] if 'TR_INIT_L' in experiment_config else 0.8,
                    scale_TR_sides=experiment_config['scale_TR_sides'] if 'scale_TR_sides' in experiment_config else True,
                    RAASP=experiment_config['RAASP'] if 'RAASP' in experiment_config else True, # random axis-aligned subspace perturbations
                    BOCK=experiment_config['BOCK'] if 'BOCK' in experiment_config else False, # whether or not to use BOCK kernel
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    min_cuda=0,  # Run on the CPU for small datasets
                    device=tkwargs['device'],  # "cpu" or "cuda"
                    dtype="float64",  # float64 or float32
                    logdir=RUN_DIR,
                )
            elif alg == 'cts-turbo' or alg == 'cts-bo':
                turbo = STurbo1(
                    f=problem,  # Handle to objective function
                    lb=lb,  # Numpy array specifying lower bounds
                    ub=ub,  # Numpy array specifying upper bounds
                    n_init=RANDOM_INIT_SAMPLES,  # Number of initial bounds from an Latin hypercube design
                    max_evals=EVAL_BUDGET,  # Maximum number of evaluations
                    batch_size=experiment_config['BATCH_SIZE'],  # How large batch size TuRBO uses
                    verbose=experiment_config['verbose'],  # Print information from each batch
                    failtol=experiment_config['failure_streak'] if 'failure_streak' in experiment_config else None,
                    rho_init=experiment_config['rho_init'] if 'rho_init' in experiment_config else 1.0,
                    rho_min=experiment_config['rho_min'] if 'rho_min' in experiment_config else 1e-2,
                    sigma_init=experiment_config['sigma_init'] if 'sigma_init' in experiment_config else 1.0,
                    n_cand=experiment_config['RAW_SAMPLES'] if 'RAW_SAMPLES' in experiment_config else None,
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    min_cuda=0,  # Run on the CPU for small datasets
                    device=tkwargs['device'],  # "cpu" or "cuda"
                    dtype="float64",  # float64 or float32
                    logdir=RUN_DIR,
                    model_space=True if 'model_space' not in experiment_config else experiment_config['model_space']
                )
        turbo.optimize()

        train_x = torch.tensor(turbo.X, **tkwargs)
        train_obj = -torch.tensor(turbo.fX, **tkwargs)

        batch_number = torch.cat(
                [torch.zeros(RANDOM_INIT_SAMPLES), torch.arange(1,  EVAL_BUDGET - RANDOM_INIT_SAMPLES+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
            ).numpy()
        os.makedirs(f'{EXP_NAME}/results/random', exist_ok=True)
        os.makedirs(f'{EXP_NAME}/results/BO', exist_ok=True)
        torch.save(batch_number,f'{EXP_NAME}/results/batch_number.pt')

        # BO
        torch.save(train_x,f'{EXP_NAME}/results/BO/BO_configs.pt')
        torch.save(train_obj,f'{EXP_NAME}/results/BO/BO_metrics.pt')
        indicator = -torch.tensor(np.minimum.accumulate(turbo.fX), **tkwargs)
        # filter to get only values after batches
        n_batches = (EVAL_BUDGET - RANDOM_INIT_SAMPLES)//BATCH_SIZE
        indices = [RANDOM_INIT_SAMPLES + x*BATCH_SIZE -1 for x in range(n_batches+1)]
        indicator = indicator[indices]
        # convert to list to conform with other scripts
        indicator = [val.squeeze() for val in indicator]
        indicator_path = f'{EXP_NAME}/results/indicator.p'
        with open(indicator_path, "wb") as file:
            pickle.dump(indicator, file)


    

if __name__ == '__main__':
    parser = common_arg_parser()
    args = parser.parse_args()
    main(args)