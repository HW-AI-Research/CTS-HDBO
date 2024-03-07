import torch
import os
import yaml
from cts.utils.parsing import common_arg_parser
from cts.morbo.run_one_replication import run_one_replication
from typing import Any, Dict
from pathlib import Path
from cts.bo_utils.helper_bo import setup_run_dir
from logging import info

BASE_SEED = 12346

def main(args):

    print(args.gpu)
    tkwargs = {
        "dtype": torch.double,
        # "dtype": torch.float,
        "device": torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu"),
    }

    with open(f'{args.exp_config}', 'r') as fp:
        experiment_config = yaml.safe_load(fp)
   
    kwargs = experiment_config.copy()
    del kwargs['TRIALS']

    BASE_EXP_NAME = args.logdir


    trials = experiment_config['TRIALS'] if 'TRIALS' in experiment_config else 1

    for trial in range(trials):
        info(f'============= Trial {trial} =============')
        if trials > 1:
            EXP_NAME = BASE_EXP_NAME + f'/trial_{trial}'
            os.makedirs(f'{EXP_NAME}')
        else:
            EXP_NAME = BASE_EXP_NAME
            
        experiment_config['EXP_NAME'] = EXP_NAME
        RUN_DIR = setup_run_dir(experiment_config, None)
        output_path = Path(RUN_DIR) / 'data.pt'

        save_callback = lambda data: torch.save(data, output_path)
        run_one_replication(
            seed=BASE_SEED+trial,
            label='morbo',
            save_callback=save_callback,
            **kwargs,
            **tkwargs
        )


    
if __name__ == '__main__':
    parser = common_arg_parser()
    args = parser.parse_args()
    main(args)