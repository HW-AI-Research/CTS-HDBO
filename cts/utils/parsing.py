from argparse import ArgumentParser, Namespace
import functools

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    parser = arg_parser()

    parser.add_argument("--run_dir", default = 'runs/smoke_test/', type = str, help="name of experiment for logging")
    parser.add_argument("--exp_config", default = 'exp_configs/smoke_test.yaml', type = str, help="name of experiment config file")
    parser.add_argument("--rnd_seed", default = 12345, type = int, help= "set random seed for experiments")
    parser.add_argument("--recovery_path", default = None, type = str, help="path to recovery files")
    parser.add_argument("--gpu", default=-1, type = int, help="gpu device number")
    parser.add_argument("--plot_dirs", nargs='+', help="list of run dirs for plotting util")
    parser.add_argument("--names", nargs='+', help="list of run names for legend in plotting util")
    parser.add_argument("--title", default=None, help="title for plot")
    parser.add_argument("--trim", action='store_true', help="whether to trim hv plot to shortest run")
    parser.add_argument("--log", action='store_true', help="whether to plot y in log scale")
    parser.add_argument("--save_dir", default = 'tmp', type = str, help="path to save figures")
    parser.add_argument("--logdir", default = 'tmp', type = str, help="path to save experiment logs")
    parser.add_argument("--algorithm", default = 'cts-turbo', type = str, help="name of algorithm to run")
    return parser

def get_unified_parser():
    """
    Define a CLI parser and parse command line arguments

    Args:
        args: command line arguments

    Returns:
        Namespace: parsed command line arguments

    """
    parser = ArgumentParser()
    required_named = parser.add_argument_group("required named arguments")
    parser.add_argument(
            "-id",
            "--input-dim",
            type=int,
            default=100,
            help="Input dimensionality",
        )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory to store results in",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to print debug messages"
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="cts-turbo",
        choices=[
            'turbo', 
            'cts-turbo',
            'baxus',
            'cts-baxus', 
            'morbo',
            'cts-morbo',
            'cts-bo', 
            'random',
            'thompson_sampling', 
            'bock', 
            'saasbo',
            'qnehvi',
        ],
        help="The algorithm"
    )

    parser.add_argument(
        "-r",
        "--num-repetitions",
        default=1,
        type=int,
        help="Number of independent repetitions of each run.",
    )

    parser.add_argument(
        "-m",
        "--max-evals",
        type=int,
        default=300,
        help="Max number of evaluations of each algorithm.",
    )
    parser.add_argument(
        "--dev",
        type=str,
        default='cpu',
        help="Device for training GP",
    )

    parser.add_argument("--gpu", default=-1, type = int, help="gpu device number")

    required_named.add_argument(
        "-f",
        "--function",
        choices=[
            "hartmann6-500d",
            "branin2-500d",
            "svm",
            "lasso-hard",
            "mopta08",
            "dtlz2-300d",
            "branincurrin-300d",
            "rover-multiobj",
        ],
        required=True,
    )

    return parser