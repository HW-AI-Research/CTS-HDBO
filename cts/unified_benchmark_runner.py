import json
import logging
import os
import sys
from datetime import datetime
from typing import List

from pathlib import Path
from argparse import Namespace
from cts.utils.parsing import get_unified_parser
import cts.main_turbo as main_turbo
import cts.main_bo as main_bo
import cts.main_morbo as main_morbo
from cts.baxus.benchmark_runner import main as main_baxus
import yaml
from logging import info

FORMAT = "%(asctime)s %(levelname)s: %(filename)s: %(message)s"
DATEFORMAT = '%m/%d/%Y %I:%M:%S %p'


def main(args) -> None:
    """
    Parse the argstring and run algorithms based on the definition.

    .. note::
        This function should not be called directly but is called by benchmark_runner.py in the project root.

    Args:
        argstring: the argument string

    Returns: Nothing

    """
    logdir = os.path.join(
        args.results_dir,
        f"{datetime.now().strftime('%d_%m_%Y')}",
        f"{args.function}",
        f"{args.algorithm}",
        f"{datetime.now().strftime('T%H-%M-%S')}",
    )
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logdir, "logging.log"),
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format=FORMAT,
        force=True,
        datefmt=DATEFORMAT
    )

    sysout_handler = logging.StreamHandler(sys.stdout)
    sysout_handler.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))
    logging.getLogger().addHandler(sysout_handler)

    info(f'log dir: {logdir}')

    args_dict = vars(args)
    with open(os.path.join(logdir, "conf.json"), "w") as f:
        f.write(json.dumps(args_dict))

    exp_cfg_basepath = Path('exp_configs')

    benchmark = args.function
    exp_cfg_path = exp_cfg_basepath / benchmark / f'{args.algorithm.upper()}.yaml'
    
    with open(f'{exp_cfg_path}', 'r') as fp:
        exp_cfg = yaml.safe_load(fp)
        new_cfg = {}
        for key, val in exp_cfg.items():
            new_key = key.replace('-', '_')
            new_cfg[new_key] = val

    if benchmark in ['branincurrin-300d', 'dtlz2-300d', 'rover-multiobj']:
        multiobj = True
    else:
        multiobj = False

    alg_arg_dict = {
        "exp_config": exp_cfg_path, 
        "logdir": logdir, 
        "gpu": args.gpu,
        "algorithm": args.algorithm,
        "multiobj": multiobj
    }

    alg_args = Namespace(**alg_arg_dict, **new_cfg)

    if 'baxus' in args.algorithm:
        with open(f'{exp_cfg_path}', 'r') as fp:
            baxus_cfg = yaml.safe_load(fp)

        # baxus_args = Namespace(**alg_arg_dict)
        baxus_arg_list = []
        for arg in baxus_cfg:
            if arg == 'adjust-initial-target-dimension' and baxus_cfg[arg]:
                baxus_arg_list.append(f'--{arg}')
            else:    
                baxus_arg_list.append(f'--{arg}')
                baxus_arg_list.append(str(baxus_cfg[arg]))
        
        baxus_arg_list.append("--results-dir")
        baxus_arg_list.append(logdir)
    

    if args.algorithm == 'turbo':
        main_turbo.main(alg_args)

    elif args.algorithm == 'cts-turbo':
        main_turbo.main(alg_args)

    elif args.algorithm == 'baxus':
        main_baxus(baxus_arg_list)

    elif args.algorithm == 'cts-baxus':
        main_baxus(baxus_arg_list)

    elif args.algorithm == 'morbo':
        main_morbo.main(alg_args)
    
    elif args.algorithm == 'cts-morbo':
        main_morbo.main(alg_args)
    
    elif args.algorithm == 'cts-bo':
        main_turbo.main(alg_args)

    elif args.algorithm == 'random':
        main_bo.main(alg_args)
        
    elif args.algorithm == 'thompson_sampling':
        main_turbo.main(alg_args)

    elif args.algorithm == 'bock':
        main_turbo.main(alg_args)

    elif args.algorithm == 'saasbo':
        main_bo.main(alg_args)

    elif args.algorithm == 'qnehvi':
        main_bo.main(alg_args)

    else:
        raise Exception('Algorithm not supported.')
    
    
    
if __name__ == "__main__":
    parser = get_unified_parser()
    args = parser.parse_args()
    main(args)