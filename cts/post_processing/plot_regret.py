import pickle
import numpy as np
from cts.utils.parsing import common_arg_parser
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch

import matplotlib.font_manager as font_manager
from matplotlib import rc
import matplotlib.pylab as plt

markevery = 100
markersize = 0

plot_kwarg_map = {
    'BAxUS': {'color': 'magenta', 'marker': "*", 'linestyle': ':', 'markersize': markersize, 'markevery':markevery},
    'CTS-BAxUS': {'color': 'magenta', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'TuRBO-1': {'color': 'blue', 'marker': "s", 'linestyle': ':', 'markersize': markersize, 'markevery':markevery},
    'CTS-TuRBO-1': {'color': 'blue', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'SAASBO': {'color': 'black', 'marker': "^", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'Sobol': {'color': 'tab:red', 'marker': "X", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'Random Search': {'color': 'tab:red', 'marker': "X", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'Random Search (Sobol)': {'color': 'tab:red', 'marker': "X", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'Global CTS': {'color': 'green', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'CTS-BO': {'color': 'green', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'MORBO': {'color': 'tab:orange', 'marker': "o", 'linestyle': ':', 'markersize': markersize, 'markevery':markevery},
    'CTS-MORBO': {'color': 'tab:orange', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
    'qNEHVI': {'color': 'tab:brown', 'marker': "o", 'linestyle': '-', 'markersize': markersize, 'markevery':markevery},
}


def main(args):

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    font_scale = 1.4
    fontdict = {"fontsize": 20*font_scale}
    font = font_manager.FontProperties(size=18*font_scale)
    # fig, axes = plt.subplots(1, 1, figsize=(24, 7*2))
    # font = font_manager.FontProperties(size=14*font_scale)
    # font = font_manager.FontProperties(size=11*font_scale)

    # fig, axes = plt.subplots(1, 1, figsize=(9, 7))
    fig, axes = plt.subplots(1, 1, figsize=(9*1.5, 7*1.5))

    print(args.plot_dirs)
    mean_ys = []
    evals = []
    for name, run_dir in zip(args.names, args.plot_dirs):

        y = None
        for trial_dir in Path(run_dir).iterdir():

            indicator_path = trial_dir / 'results/BO/BO_metrics.pt'
            if not indicator_path.is_file():
                print(f'No BO_metrics.pt file at {indicator_path}')
                continue
            new_y = torch.load(indicator_path, map_location='cpu').T
            print(new_y.shape)
            if y is not None and new_y.shape[-1] > len(y[-1]):
                print(indicator_path)
                # print(new_y)
                new_y = new_y[:, -len(y[-1]):]
            if y is None:
                y = new_y
                # print(y.shape)
            else:
                y = torch.cat([y, new_y], axis=0)
        
        y *= -1 # negate
        if "hartmann" in args.title.lower():
            y = y - -3.32237 # optimal value for hartmann6
        elif "branin" in args.title.lower():
            y = y - 0.397887 # optimal value for branin
        n = y.shape[0]
        if 'swimmer' or 'robot' in args.title.lower():
            incumbents = -y.cummin(dim=1).values
        else:
            incumbents = y.cummin(dim=1).values
            incumbents = incumbents.clamp(min=1e-3)
        if args.log:
            log_y = torch.log10(incumbents)
            mean_log_y = log_y.mean(axis=0)
            std_err_log_y = log_y.std(axis=0) / torch.sqrt(torch.tensor(n))
            mean_y = 10 ** mean_log_y
            lower = 10 ** (mean_log_y - 1.96 * std_err_log_y)
            upper = 10 ** (mean_log_y + 1.96 * std_err_log_y)
        else:
            mean_y = incumbents.mean(axis=0)
            std_err = incumbents.std(axis=0) / torch.sqrt(torch.tensor(n))
            lower = mean_y - 1.96 * std_err
            upper = mean_y + 1.96 * std_err
        # if args.log:
        #     std_err = 10**std_err

        print(f'{name} final vals: {incumbents[:,-1]}')
        
        x = np.arange(len(mean_y)) + 1
        evals.append(x[-1])
        # axes.plot(x, mean_y, label=f'{name}: n = {n}', linewidth=3)
        if name in plot_kwarg_map:
        # if False:
            axes.plot(x, mean_y, label=f'{name}', linewidth=3, **plot_kwarg_map[name])
            axes.fill_between(x, lower, upper, alpha=0.2, color=plot_kwarg_map[name]['color'])
        else:
            axes.plot(x, mean_y, label=f'{name}', linewidth=3)
            axes.fill_between(x, lower, upper, alpha=0.2)
        

        mean_ys.append(mean_y)
    
    # if args.trim:
    #     plt.xlim([num_init, min(evals)])

    #     min_hv = min([min(hv) for hv in mean_ys])
    #     max_hv = max([max(hv[:min(evals)]) for hv in mean_ys])

    #     plt.ylim([min_hv, max_hv])

    if args.log:
        axes.set_yscale('log')

    axes.grid(True, 'major', axis='both', linestyle='--', linewidth=1.5)
    axes.grid(True, 'minor', axis='y', linestyle='dotted', linewidth=1.5)
    # font sizes
    axes.tick_params(width=1.5, axis='both', which='major', labelsize=16*font_scale)
    axes.tick_params(width=1.5, axis='both', which='minor', labelsize=16*font_scale)

    # plt.ylim([0.9, 110])
    # plt.ylim([5e-4, 110])
    # plt.legend(prop=font, framealpha=0.4)
    # axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #       fancybox=True, shadow=True, ncol=2)
    # axes.legend(prop=font, loc='upper right',
    #       fancybox=True, shadow=True, ncol=2)
    # axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #       fancybox=True, shadow=True, ncol=3)
    # axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #       fancybox=True, shadow=True, ncol=4)

    # axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #       fancybox=True, shadow=True, ncol=6)
    # axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #       fancybox=True, shadow=True, ncol=7)

    # axes.legend(prop=font,
    #       fancybox=True, shadow=True, ncol=5)

    axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

    # axes.legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #       fancybox=True, shadow=True, ncol=1)

    plt.xlabel('Number of Evaluations', **fontdict)
    
    if 'swimmer' or 'robot' in args.title.lower():
        plt.ylabel('Reward', **fontdict)
    else:    
        plt.ylabel('Regret', **fontdict)
    # plt.ylabel('Function Value', **fontdict)
    if args.title is not None:
        # plt.title(args.title, fontname='Times New Roman', fontsize=22*font_scale)
        plt.title(args.title, fontsize=22*font_scale)
    plt.tight_layout()
    plot_save_dir = Path(f'./plots/{args.save_dir}')
    plot_save_dir.mkdir(exist_ok=True, parents=True)
    plot_save_path = plot_save_dir / 'test_regret.png'
    print(plot_save_path)
    plt.savefig(plot_save_path)

if __name__ == '__main__':
    parser = common_arg_parser()
    args = parser.parse_args()
    main(args)