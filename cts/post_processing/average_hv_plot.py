# import pickle
# import numpy as np
# from cts.utils.parsing import common_arg_parser
# import matplotlib.pyplot as plt
# from pathlib import Path
# import yaml
# import torch

# def main(args):
    
#     fig, axes = plt.subplots(1, 1, figsize=(9, 7))
#     print(args.plot_dirs)
#     mean_hvs = []
#     evals = []
#     for name, run_dir in zip(args.names, args.plot_dirs):
#         #print(run_dir)
#         hvs = []

#         # get the batch size so we can plot evaluations on x-axis
#         config_dir = Path(run_dir) / 'trial_1/configs/experiment_config.yaml'
#         with open(config_dir, 'r') as fp:
#             experiment_config = yaml.safe_load(fp)

#         batch_size = experiment_config['BATCH_SIZE']
#         num_init = experiment_config['RANDOM_INIT_SAMPLES']
#         #num_init = 0

#         for trial_dir in Path(run_dir).iterdir():
#             indicator_path = trial_dir / 'results/hypervolume.p'
#             if not indicator_path.is_file():
#                 indicator_path = trial_dir / 'results/indicator.p'
#             if not indicator_path.is_file():
#                 continue

#             with open(indicator_path, "rb") as file:
#                 hv_list = pickle.load(file)

#                 hv_list = [torch.tensor(hv) for hv in hv_list]
#                 hv_list = [hv.cpu() for hv in hv_list]
#                 hv = np.asarray(hv_list)
#                 hvs.append(hv)    

#             # try:
#             #     hv_path = trial_dir / 'results/hypervolume.p'
#             #     # hv_path = trial_dir / 'results/indicator.p'


#             # # if hv_path.exists():
#             #     with open(hv_path, "rb") as file:
#             #         # hv = np.asarray(pickle.load(file))
#             #         # hvs.append(hv)
#             #         hv_list = pickle.load(file)
#             #         # print(f'{name}: {len(hv_list)}')
#             #         # print(type(hv_list))
#             #         # print(type(hv_list[0]))

#             #         if isinstance(hv_list[0], float): # hack for plotting results from danila with first instance float
#             #             hv_list_tail = hv_list[1:]
#             #             # print(hv_list_tail)
#             #             hv_list = [torch.tensor(hv_list[0])]
#             #             # print(hv_list)
#             #             hv_list.extend(hv_list_tail)
#             #             # print(hv_list)
#             #             # print(hv_list)
#             #             # print('is float')
#             #             hv_list = [hv.cpu() for hv in hv_list]
#             #             hv = np.asarray(hv_list)
#             #             hvs.append(hv)    
#             #             # hv = np.asarray(hv_list)
#             #             # hvs.append(hv)
#             #         else:
#             #             hv_list = [hv.cpu() for hv in hv_list]
#             #             hv = np.asarray(hv_list)
#             #             hvs.append(hv)    
                    

#             # except Exception as error:
#             #     # print(error)
#             #     hv_path = trial_dir / 'results/indicator.p'
#             #     # if hv_path.exists():
#             #     with open(hv_path, "rb") as file:
#             #         hv_list = pickle.load(file)
#             #         # print(f'{name}: {len(hv_list)}')
#             #         # print(hv_list)
#             #         hv_list = [hv.cpu() for hv in hv_list]
#             #         hv = np.asarray(hv_list)
#             #         hvs.append(hv)

#         max_length = max(len(hv) for hv in hvs)
#         min_length = min(len(hv) for hv in hvs)
#         #hvs = [hv[:min_length] for hv in hvs]
#         hvs = [hv for hv in hvs if len(hv) == max_length]

#         hvs = np.array(hvs)
#         print(f'{name} final HVs: {hvs[:,-1]}')
#         mean_hv = np.mean(hvs, axis=0)
#         n = hvs.shape[0]
#         std_hv = np.std(hvs, axis=0)
#         st_err_hv = std_hv/np.sqrt(n)
#         x = np.linspace(0, (len(mean_hv)-1)*batch_size, len(mean_hv)) + num_init
#         evals.append(int(max(x)))
#         # if name.lower() == "baxus":
#         #     for i, hv in enumerate(hvs):
#         #         axes.plot(x, hv, label=f'{name}: n = {n}' if i == 0 else None, color="red")            
#         # else:
#         #     axes.plot(x, mean_hv, label=f'{name}: n = {n}')
#         #     axes.fill_between(x, mean_hv-st_err_hv,mean_hv+st_err_hv, alpha=0.2)
#         axes.plot(x, mean_hv, label=f'{name}: n = {n}')
#         axes.fill_between(x, mean_hv-st_err_hv,mean_hv+st_err_hv, alpha=0.2)

#         mean_hvs.append(mean_hv)
    
#     if args.trim:
#         plt.xlim([num_init, min(evals)])

#         min_hv = min([min(hv) for hv in mean_hvs])
#         max_hv = max([max(hv[:min(evals)]) for hv in mean_hvs])

#         plt.ylim([min_hv, max_hv])

#     plt.legend()
#     plt.xlabel('Number of Evaluations')
#     plt.ylabel('Hypervolume')
#     if args.title is not None:
#         plt.title(args.title)

#     plt.savefig(f'./plots/test_hv_curve_morbo.png')


# if __name__ == '__main__':
#     parser = common_arg_parser()
#     args = parser.parse_args()
#     main(args)

import pickle
import numpy as np
from cts.utils.parsing import common_arg_parser
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch

def main(args):
    
    fig, axes = plt.subplots(1, 1, figsize=(9, 7))
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

        batch_size = experiment_config['BATCH_SIZE']
        # batch_size = 50
        num_init = experiment_config['RANDOM_INIT_SAMPLES']
        # num_init = 10

        for trial_dir in Path(run_dir).iterdir():
            #print(trial_dir)
            try:
                hv_path = trial_dir / 'results/hypervolume.p'


            # if hv_path.exists():
                with open(hv_path, "rb") as file:
                    # hv = np.asarray(pickle.load(file))
                    # hvs.append(hv)
                    hv_list = pickle.load(file)
                    # print(f'{name}: {len(hv_list)}')
                    # print(type(hv_list))
                    # print(type(hv_list[0]))

                    if isinstance(hv_list[0], float): # hack for plotting results from danila with first instance float
                        hv_list_tail = hv_list[1:]
                        # print(hv_list_tail)
                        hv_list = [torch.tensor(hv_list[0])]
                        # print(hv_list)
                        hv_list.extend(hv_list_tail)
                        # print(hv_list)
                        # print(hv_list)
                        # print('is float')
                        hv_list = [hv.cpu() for hv in hv_list]
                        hv = np.asarray(hv_list)
                        hvs.append(hv)    
                        # hv = np.asarray(hv_list)
                        # hvs.append(hv)
                    else:
                        hv_list = [hv.cpu() for hv in hv_list]
                        hv = np.asarray(hv_list)
                        hvs.append(hv)    
                    

            except Exception as error:
                # print(error)
                hv_path = trial_dir / 'results/indicator.p'
                # if hv_path.exists():
                with open(hv_path, "rb") as file:
                    hv_list = pickle.load(file)
                    # print(f'{name}: {len(hv_list)}')
                    # print(hv_list)
                    hv_list = [hv.cpu() if type(hv) != float else hv for hv in hv_list]
                    # hv_list = 
                    hv = np.asarray(hv_list)
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
        x = np.linspace(0, (len(mean_hv)-1)*batch_size, len(mean_hv)) + num_init
        evals.append(int(max(x)))
        axes.plot(x, mean_hv, label=f'{name}: n = {n}')
        axes.fill_between(x, mean_hv-st_err_hv,mean_hv+st_err_hv, alpha=0.2)

        mean_hvs.append(mean_hv)
    
    if args.trim:
        plt.xlim([num_init, min(evals)])

        min_hv = min([min(hv) for hv in mean_hvs])
        max_hv = max([max(hv[:min(evals)]) for hv in mean_hvs])

        plt.ylim([min_hv, max_hv])

    plt.legend()
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Hypervolume')
    if args.title is not None:
        plt.title(args.title)

    plt.savefig(f'./runs/test_hv_curve_morbo.png')


if __name__ == '__main__':
    parser = common_arg_parser()
    args = parser.parse_args()
    main(args)