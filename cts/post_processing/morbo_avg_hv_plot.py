import pickle
import numpy as np
from cts.utils.parsing import common_arg_parser
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
import json

import matplotlib.font_manager as font_manager
from matplotlib import rc

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


markevery = 100
markersize = 0

plot_kwarg_map = {
    'BAxUS': {'color': 'magenta', 'marker': "*", 'linestyle': ':', 'markersize': markersize, 'markevery':markevery},
    'CTS-BAxUS': {'color': 'magenta', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'TuRBO-1': {'color': 'blue', 'marker': "s", 'linestyle': ':', 'markersize': markersize, 'markevery':markevery},
    'CTS-TuRBO-1': {'color': 'blue', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'SAASBO': {'color': 'black', 'marker': "^", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'Sobol': {'color': 'tab:red', 'marker': "X", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'Global CTS': {'color': 'green', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'MORBO': {'color': 'tab:orange', 'marker': "o", 'linestyle': ':', 'markersize': markersize, 'markevery':markevery},
    # 'MORBO (Baseline)': {'color': 'tab:orange', 'marker': "o", 'linestyle': ':', 'markersize': markersize, 'markevery':markevery},
    'CTS-MORBO': {'color': 'tab:orange', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'qNEHVI': {'color': 'tab:brown', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
}

def main(args):
    

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    font_scale = 1.4
    # fontdict = {'fontname':'Times New Roman', "fontsize": 20*font_scale}
    fontdict = {"fontsize": 20*font_scale}
    # font = font_manager.FontProperties(family='Times New Roman',
    #                             #    weight='bold',
    #                                style='normal', size=18*font_scale)
    # font = font_manager.FontProperties(family='serif', serif=['Computer Modern'],
    #                                style='normal', size=18*font_scale)

    font = font_manager.FontProperties(size=18*font_scale)

    fig, axes = plt.subplots(1, 1, figsize=(9, 7))
    # fig, axes = plt.subplots(1, 1, figsize=(24, 7*2)) # for getting shared legend
    print(args.plot_dirs)
    mean_hvs = []
    evals = []
    for name, run_dir in zip(args.names, args.plot_dirs):
        #print(run_dir)
        hvs = []

        # get the batch size so we can plot evaluations on x-axis
        config_dir = Path(run_dir) / 'trial_0/configs/experiment_config.yaml'
        with open(config_dir, 'r') as fp:
            experiment_config = yaml.safe_load(fp)

        # batch_size = experiment_config['BATCH_SIZE']
        batch_size = 50
        # num_init = experiment_config['RANDOM_INIT_SAMPLES']
        num_init = 200
        #num_init = 0

        test_data_path = Path(run_dir) / 'trial_0/data.pt'
        if test_data_path.exists():

            for trial_dir in Path(run_dir).iterdir():
                indicator_path = trial_dir / 'data.pt'
                if indicator_path.exists():
                    data = torch.load(indicator_path)
                    hv = np.asanyarray(data['true_hv'])
                    # init_hv = get_initial_hv(trial_dir, experiment_config)
                    # hv = np.asarray(data['all_hvs'])
                    # init_hv = get_initial_hv(trial_dir, experiment_config)
                    # hv = np.concatenate([init_hv, hv])
                    hvs.append(hv)    

        else:
            for trial_dir in Path(run_dir).iterdir():
                indicator_path = trial_dir / 'results/indicator.p'
                # init_hv = get_initial_hv(trial_dir, experiment_config)
                with open (indicator_path, 'rb') as f:
                    hv = pickle.load(f)
                # exclude first hv (from random init) to be consistent with morbo logging
                # hv = hv[1:]
                # hv = np.concatenate([init_hv, hv])
                hvs.append(hv)    

        max_length = max(len(hv) for hv in hvs)
        min_length = min(len(hv) for hv in hvs)
        #hvs = [hv[:min_length] for hv in hvs]
        hvs = [hv for hv in hvs if len(hv) == max_length]

        hvs = np.array(hvs)
        
        print(f'{name} final HVs: {hvs[:,-1]}')
        mean_hv = np.mean(hvs, axis=0)
        n = hvs.shape[0]
        std_hv = np.std(hvs, axis=0)
        st_err_hv = std_hv/np.sqrt(n)
        x = np.linspace(0, (len(mean_hv)-1)*batch_size, len(mean_hv)) + num_init #+ batch_size
        evals.append(int(max(x)))
        # axes.plot(x, mean_hv, label=f'{name}: n = {n}')
        # axes.plot(x, mean_hv, label=f'{name}', linewidth=3)
        # axes.fill_between(x, mean_hv-st_err_hv,mean_hv+st_err_hv, alpha=0.2)
        lower = mean_hv-1.96 * st_err_hv
        upper = mean_hv+1.96 * st_err_hv
        if name in plot_kwarg_map:
        # if False:
            axes.plot(x, mean_hv, label=f'{name}', linewidth=3, **plot_kwarg_map[name])
            axes.fill_between(x, lower, upper, alpha=0.2, color=plot_kwarg_map[name]['color'])
        else:
            axes.plot(x, mean_hv, label=f'{name}', linewidth=3)
            axes.fill_between(x, lower, upper, alpha=0.2)

        mean_hvs.append(mean_hv)
    
    if args.trim:
        plt.xlim([num_init, min(evals)])

        min_hv = min([min(hv) for hv in mean_hvs])
        max_hv = max([max(hv[:min(evals)]) for hv in mean_hvs])

        plt.ylim([min_hv, max_hv])


    axes.grid(True, 'major', axis='both', linestyle='--', linewidth=1.5)
    axes.grid(True, 'minor', axis='y', linestyle='dotted', linewidth=1.5)
    # font sizes
    axes.tick_params(width=1.5, axis='both', which='major', labelsize=16*font_scale)
    axes.tick_params(width=1.5, axis='both', which='minor', labelsize=16*font_scale)

    # plt.ylim([0.9, 110])
    # plt.ylim([5e-4, 110])
    plt.legend(prop=font, framealpha=0.3)

    # axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #       fancybox=True, shadow=True, ncol=2)
    plt.xlabel('Number of Evaluations', **fontdict)
    plt.ylabel('Hypervolume', **fontdict)
    # plt.ylabel('Function Value')
    if args.title is not None:
        # plt.title(args.title, fontname='Times New Roman', fontsize=22*font_scale)
        plt.title(args.title, fontsize=22*font_scale)
    plt.tight_layout()

    plot_save_dir = Path(f'./plots/{args.save_dir}')
    plot_save_dir.mkdir(exist_ok=True, parents=True)
    plot_save_path = plot_save_dir / 'morbo_test_hv.png'
    plt.savefig(plot_save_path)


def get_initial_hv(trial_path, exp_config):

    ref = exp_config['max_reference_point']
    y_init = torch.load(trial_path / 'initial_obj.pt')

    pareto_mask = is_non_dominated(y_init)
    partitioning = DominatedPartitioning(
        ref_point=torch.tensor(ref).to(y_init), Y=y_init[pareto_mask]
    )

    init_hv = partitioning.compute_hypervolume().item()

    return np.array([init_hv])



if __name__ == '__main__':
    parser = common_arg_parser()
    args = parser.parse_args()
    main(args)