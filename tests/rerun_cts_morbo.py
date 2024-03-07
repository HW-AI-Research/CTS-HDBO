from cts.unified_benchmark_runner import main
from argparse import Namespace
from logging import info

multi_obj_algs = [
    'cts-morbo',
]

multi_obj_benchmarks = [
    "dtlz2-300d",
    "branincurrin-300d",
    "rover-multiobj",
]

all_algs = [multi_obj_algs]
all_benchmarks = [multi_obj_benchmarks]

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