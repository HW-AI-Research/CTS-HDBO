# CTS
Install in a fresh environment: 
```
pip install -e .
```

## Rerun CTS-MORBO results

Please run the following three commands in parallel using different terminals (or screen windows). 

First please check which gpus, if any, are busy on the server with the following command:
```
nvidia-smi -l 1
```
The servers you are working with have 8 gpus. You can select 3 that are free and then modify the gpu argument in the following commands accordingly.

The commands to run CTS-MORBO for the three multi-objective benchmarks are:

```
python cts/unified_benchmark_runner.py -v --num-repetitions 10 --algorithm cts-morbo --function dtlz2-300d --gpu 0
```

```
python cts/unified_benchmark_runner.py -v --num-repetitions 10 --algorithm cts-morbo --function branincurrin-300d --gpu 1
```

```
python cts/unified_benchmark_runner.py -v --num-repetitions 10 --algorithm cts-morbo --function rover-multiobj --gpu 2
```

I have set the max_centers to 10 and reverted sigma_init to the original values reported in our paper.

We need to re-run the results to ensure there was no bug.

You will find the results in the `results` directory.

Reproduce results:
```
python tests/run_every_benchmark.py
```

NOTE: If you get a memory error while running CTS-MORBO, go to the config yaml file (e.g., `exp_configs/dtlz2-300d/CTS-MORBO.yaml`), and uncomment the line that sets a limit on the number of points that can be used a centers of perturbation:
```
max_centers: 30
```

You will find the results in the `results` directory.

## ToDO

- [ ] Pass sigma init to cts-baxus
- [ ] Unify result handling
- [ ] Double check budget in all configs
- [ ] Automatically run `save_result_tensor` after baxus algs finish
