from cts.unified_benchmark_runner import main
from argparse import Namespace
from logging import info

single_obj_algs = [
    'random', 
    'saasbo',
    'cts-baxus',
    'baxus',
    'turbo', 
    'cts-turbo',
    'cts-bo', 
    'thompson_sampling',
    'bock', 
]

multi_obj_algs = [
    'random',
    'morbo',
    'cts-morbo',
    'qnehvi',
]

single_obj_benchmarks = [
    "hartmann6-500d",
    "branin2-500d",
    "svm",
    "lasso-hard",
    "mopta08",
]

multi_obj_benchmarks = [
    "dtlz2-300d",
    "branincurrin-300d",
    "rover-multiobj",
]

all_algs = [single_obj_algs, multi_obj_algs]
all_benchmarks = [single_obj_benchmarks, multi_obj_benchmarks]

# all_algs = [single_obj_algs]
# all_benchmarks = [single_obj_benchmarks]

# all_algs = [multi_obj_algs]
# all_benchmarks = [multi_obj_benchmarks]

for algs, benchmarks in zip(all_algs, all_benchmarks):
    for alg in algs:
        for benchmark in benchmarks:
            info(f'{alg} : {benchmark}')
            try:
                benchmark_runner_arg_dict = {
                    'results_dir': 'results',
                    'verbose': True,
                    'num_repetitions': 1,
                    "algorithm": alg, 
                    'function': benchmark,
                }
                benchmark_runner_args = Namespace(**benchmark_runner_arg_dict)
                main(benchmark_runner_args)
                print('')
            except KeyboardInterrupt:
                print('')
                continue