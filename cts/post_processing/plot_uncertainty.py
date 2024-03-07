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
import pandas as pd
from matplotlib.transforms import Affine2D
from itertools import cycle
def main(args):

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    font_scale = 1.4
    fontdict = {"fontsize": 20*font_scale}

    font = font_manager.FontProperties(size=18*font_scale)

    # fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    print(args.plot_dirs)
    
    # shift_magnitude = 3.0
    shift_magnitude = 10.0
    shift = -shift_magnitude
    
    lines = [":","--","-."]
    linecycler = cycle(lines)

    for name, sweep_dir in zip(args.names, args.plot_dirs):
        #print(run_dir)
    
        dim_dir = Path(sweep_dir)
        dims = []
        
        pert_mag_mean = []
        pert_mag_std = []
        
        ss_pert_mag_mean = []
        ss_pert_mag_std = []
        
        uncertainty_mean = []
        uncertainty_std = []
        
        for run_dir in dim_dir.iterdir():
            # get dim
            config_dir = Path(run_dir) / 'trial_1/configs/experiment_config.yaml'
            with open(config_dir, 'r') as fp:
                experiment_config = yaml.safe_load(fp)

            # dim = experiment_config['dim']
            dims.append(experiment_config['dim'])
            
            dim_data = {
                'pert_mag': [],
                'ss_pert_mag': [],
                'uncertainty': []
            }
            # gather data
            for trial_dir in Path(run_dir).iterdir():
                csv_dir = trial_dir / 'uncertainty.csv'
                df = pd.read_csv(csv_dir)
                for key in dim_data:
                    dim_data[key].extend(list(df[key]))
            
            pert_mag = np.array(dim_data['pert_mag'])
            pert_mag_mean.append(pert_mag.mean())
            pert_mag_std.append(pert_mag.std())
            
            ss_pert_mag = np.array(dim_data['ss_pert_mag'])
            ss_pert_mag_mean.append(ss_pert_mag.mean())
            ss_pert_mag_std.append(ss_pert_mag.std())
            
            uncertainty = np.array(dim_data['uncertainty'])
            uncertainty_mean.append(uncertainty.mean())
            uncertainty_std.append(uncertainty.std())
            

        # axes[0].plot(dims, mean_y, label=f'{name}', linewidth=3)
        tf = Affine2D().translate(shift, 0.0)
        
        line = next(linecycler)
        l0 = axes[0].errorbar(dims, pert_mag_mean, yerr=pert_mag_std, marker="o", linestyle=line, linewidth=3, capsize=10, capthick=3, label=f'{name}', transform=tf + axes[0].transData)
        l2 = axes[2].errorbar(dims, ss_pert_mag_mean, yerr=ss_pert_mag_std, marker="o", linestyle=line, linewidth=3, capsize=10, capthick=3, label=f'{name}', transform=tf + axes[2].transData)
        l1 = axes[1].errorbar(dims, uncertainty_mean, yerr=uncertainty_std, marker="o", linestyle=line, linewidth=3, capsize=10, capthick=3, label=f'{name}', transform=tf + axes[1].transData)
        
        axes[0].set_ylabel('Distance from Incumbent', **fontdict)
        axes[2].set_ylabel('Active Subspace Perturbation', **fontdict)
        axes[1].set_ylabel('GP Prediction Uncertainty', **fontdict)
        shift += shift_magnitude

    for ax in axes:
        ax.grid(True, 'major', axis='both', linestyle='--', linewidth=1.5)
        ax.grid(True, 'minor', axis='y', linestyle='dotted', linewidth=1.5)
        # font sizes
        ax.tick_params(width=1.5, axis='both', which='major', labelsize=16*font_scale)
        ax.tick_params(width=1.5, axis='both', which='minor', labelsize=16*font_scale)
        # ax.set_xticks(dims)
        ax.set_xticks(dims[2:])

        ax.set_xlabel('Input Dimensions', **fontdict)

    axes[2].axhline(0, linestyle='-', color='k', linewidth=1.5)
    # plt.ylim([0.9, 110])
    # plt.ylim([5e-4, 110])
    axes[1].legend(prop=font, loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=3)
    # fig.legend(prop=font, loc='lower center', bbox_to_anchor=(0.5, -0.25),
    #       fancybox=True, shadow=True, ncol=3)
    # fig.legend((l0, l1, l2), prop=font,
    #       fancybox=True, shadow=True, ncol=3)
    
    # plt.legend(prop=font, framealpha=0.4, )
    # plt.xlabel('Number of Evaluations', **fontdict)
    # plt.ylabel('Regret', **fontdict)
    # # plt.ylabel('Function Value', **fontdict)
    # if args.title is not None:
    #     # plt.title(args.title, fontname='Times New Roman', fontsize=22*font_scale)
    #     plt.title(args.title, fontsize=22*font_scale)
    # plt.subplots_adjust(wspace=0.1)
    # plt.tight_layout(h_pad=0.01)
    plt.tight_layout(w_pad=6)
    plt.savefig(f'./plots/test_uncertainty.png')

if __name__ == '__main__':
    parser = common_arg_parser()
    args = parser.parse_args()
    main(args)