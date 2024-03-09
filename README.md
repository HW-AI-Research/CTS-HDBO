# CTS
Install this package and its dependencies: 
```
pip install -e .
```
Run CTS-TuRBO on the Branin-500D function:
```
python cts/unified_benchmark_runner.py -a cts-turbo -f branin2-500d
```

Reproduce all results:
```
python tests/run_every_benchmark.py
```

You will find the results in the `results` directory.