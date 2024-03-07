import torch
from botorch import settings
import os
import time
import yaml
import numpy as np

from cts.utils.parsing import common_arg_parser
from cts.bo_utils.helper_bo import (
    save_intermediate, setup_run_dir, log_progress, 
    save_indicator, compute_indicator
)

from cts.bo_utils.essential_bo import ACQ_FNS, MODEL_FIT_FNS
from cts.baxus.util.parsing import fun_mapper, benchmark_loader
from cts.morbo.utils import get_multiobj_fn
from cts.morbo.benchmark_function import BenchmarkFunction
from cts.bo_utils.essential_bo import SobolSampler
from logging import info, debug

def get_benchmark_fn(evalfn, dim, multiobj, ref_pt, tkwargs):
    if multiobj:
        max_reference_point = ref_pt
        f, num_outputs, bounds, num_objectives, _, _ = get_multiobj_fn(evalfn, dim, max_reference_point, tkwargs)
        problem = BenchmarkFunction(
            base_f=f,
            num_outputs=num_outputs,
            ref_point=torch.tensor(max_reference_point, **tkwargs),
            dim=dim,
            tkwargs=tkwargs,
            negate=True,
            observation_noise_std=None,
            observation_noise_bias=None,
        )
    else:
        benchmark_loader(evalfn, args=None)
        problem = fun_mapper()[evalfn](dim=dim, noise_std=0.0)
        
        lb = problem.lb_vec
        ub = problem.ub_vec
        bounds = torch.tensor(np.array([lb, ub]), **tkwargs)
        num_objectives = 1
    
    return problem, bounds, num_objectives

def main(args):

    with open(f'{args.exp_config}', 'r') as fp:
        experiment_config = yaml.safe_load(fp)

    assert args.algorithm in ['saasbo', 'random', 'qnehvi']
    alg = args.algorithm

    multiobj = args.multiobj
    if multiobj:
        ref_pt = experiment_config['max_reference_point']
    else:
        ref_pt = None

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu"),
    }

    problem, bounds, num_objectives = get_benchmark_fn(evalfn=args.function, dim=args.input_dim, multiobj=multiobj, ref_pt=ref_pt, tkwargs=tkwargs)
    negate = True if not multiobj else False
           
    BASE_EXP_NAME = args.logdir

    EVAL_BUDGET = args.max_evals
    RANDOM_INIT_SAMPLES = args.n_init

    random_sampler = SobolSampler(dim=args.input_dim, bounds=bounds)
    
    acq_cfg = {'BATCH_SIZE': experiment_config['BATCH_SIZE']}
    if alg != 'random':
        acq_cfg['NUM_RESTARTS'] = experiment_config['NUM_RESTARTS']
        acq_cfg['RAW_SAMPLES'] = experiment_config['RAW_SAMPLES']

    model_cfg = {}
    if alg == 'saasbo':
        model_cfg['MCMC_WARMUP_STEPS'] = experiment_config['MCMC_WARMUP_STEPS']
        model_cfg['MCMC_NUM_SAMPLES'] = experiment_config['MCMC_NUM_SAMPLES']
        model_cfg['MCMC_THINNING'] = experiment_config['MCMC_THINNING']
    
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
        train_x, train_obj = random_sampler.sample_and_eval(n=RANDOM_INIT_SAMPLES, problem=problem)

        if negate:
            train_obj *= -1
        
        train_x = train_x.to(**tkwargs)
        train_obj = torch.tensor(train_obj, **tkwargs)
        if len(train_obj.shape) == 1:
            train_obj = train_obj.unsqueeze(-1)
        
        # save initial samples
        torch.save(train_x, f'{EXP_NAME}/initial_x.pt')
        torch.save(train_obj, f'{EXP_NAME}/initial_obj.pt')

        eval_counter = RANDOM_INIT_SAMPLES
        iteration = 1
        
        indicator = compute_indicator(problem, train_obj, num_objectives)

        # indicator logging
        indicators = [indicator]

        fit_model_fn = MODEL_FIT_FNS[experiment_config['MODEL']] if alg != 'random' else None
        acquisition_fn = ACQ_FNS[experiment_config['ACQ_FN']]  if alg != 'random' else None

        # run BayesOpt until run out of budget
        while eval_counter < EVAL_BUDGET:
            t0 = time.monotonic()

            # fit model with data observed so far...
            model = fit_model_fn(train_x, train_obj, bounds, model_cfg) if alg != 'random' else None
            model_fit_time = time.monotonic() - t0
            debug(f'Model fit time: {model_fit_time}')
            
            # get new observations
            if alg != 'random':
                # optimize acquisition functions and get new observations
                new_x, new_obj = acquisition_fn(model, train_x, problem, bounds, acq_cfg)
            else:
                new_x, new_obj = random_sampler.sample_and_eval(n=acq_cfg['BATCH_SIZE'], problem=problem)

            if negate:
                new_obj *= -1
            
            # log time
            t1 = time.monotonic()
            elapsed_time = t1-t0
            acquisition_time = elapsed_time - model_fit_time
            debug(f'Acquisition time: {acquisition_time}')

            # get types in order
            new_x = new_x.to(**tkwargs)
            new_obj = torch.tensor(new_obj, **tkwargs)
            if len(new_obj.shape) == 0:
                new_obj = new_obj.unsqueeze(-1)
            if len(new_obj.shape) == 1:
                new_obj = new_obj.unsqueeze(-1)   
            
            # update expended budget
            eval_counter += new_obj.shape[0]
            
            # update training points
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # log progress
            indicators = log_progress(problem, num_objectives, iteration, elapsed_time, indicators, train_obj, alg)
            
            # write dataset to file
            save_intermediate(train_x, train_obj, RUN_DIR)
            save_indicator(indicators, RUN_DIR)
            iteration += 1

        # clean up
        batch_number = torch.cat(
                [torch.zeros(RANDOM_INIT_SAMPLES), torch.arange(1,  EVAL_BUDGET - RANDOM_INIT_SAMPLES+1).repeat(acq_cfg['BATCH_SIZE'], 1).t().reshape(-1)]
            ).numpy()
        os.makedirs(f'{EXP_NAME}/results/BO', exist_ok=True)
        torch.save(batch_number,f'{EXP_NAME}/results/batch_number.pt')

        # BO
        torch.save(train_x,f'{EXP_NAME}/results/BO/BO_configs.pt')
        torch.save(train_obj,f'{EXP_NAME}/results/BO/BO_metrics.pt')


    

if __name__ == '__main__':
    parser = common_arg_parser()
    args = parser.parse_args()
    main(args)