from pathlib import Path
import argparse
import pandas as pd
import torch
import yaml
import pickle

def save_result_tensor(run_dir, n_init, budget):
    # we will assume run_dir has form like results/06_09_2023/lasso-hard_in_dim_1000_t_dim10_n_init_100

    # need to build directory structure to match the plotting script
    run_path = Path(run_dir)
    export_path = run_path / f'export_{run_path.stem}'
    export_path.mkdir(exist_ok=True)

    repetition_paths = run_path.glob('**/repetition*')
    for i, rep_path in enumerate(repetition_paths):
        csv_path = rep_path.parent / f'repet_{i}.csv.xz'
        if not csv_path.is_file():
            continue

        print(rep_path)
        trial_path = export_path / f'trial_{i}'
        trial_result_path = trial_path / 'results' / 'BO'
        trial_result_path.mkdir(parents=True, exist_ok=True)

        save_config_file(trial_path, n_init)
        
        df = pd.read_csv(csv_path)
        y = -torch.tensor(df['y_raw']).unsqueeze(-1)
        while y.shape[0] < budget:
            y = torch.cat([y, y[-1].unsqueeze(-1)], dim=0)
        print(y.shape)
        torch.save(y, trial_result_path / 'BO_metrics.pt')
        save_indicator(y, trial_path, n_init, budget)

def save_indicator(y: torch.Tensor, trial_path: Path, n_init: int, budget: int):
    incumbents, _ = torch.cummax(y, dim=0)
    incumbents = incumbents[n_init-1:].squeeze()
    indicator = [y_i for y_i in incumbents]
    # fill end of list to comply with plotting script
    while len(indicator) < (budget - n_init + 1):
        indicator.append(indicator[-1])
    hvs_path = trial_path / 'results' / 'indicator.p'
    with open(hvs_path, "wb") as file:
        pickle.dump(indicator, file)

def save_config_file(trial_path: Path, n_init: int):
    config_path = trial_path / 'configs' 
    config_path.mkdir(exist_ok=True)
    config_dict = {'BATCH_SIZE': 1, 'RANDOM_INIT_SAMPLES': n_init}
    with open(config_path / 'experiment_config.yaml', 'w') as fp:
        yaml.dump(config_dict, fp)

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_dir", default = 'runs/smoke_test/', type = str, help="name of experiment for logging")
    parser.add_argument("--n_init", default = 10, type = int, help="number of initial samples")
    parser.add_argument("--budget", default = 500, type = int, help="search budget")

    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    save_result_tensor(args.run_dir, args.n_init, args.budget)