import numpy as np
from foreman import Foreman


def expand_experiment(dict):
    exp = {}
    for k, v in dict.items():
        expand = False
        if isinstance(v, list):
            if isinstance(v[0], int):
                exp[k] = list(np.linspace(v[0], v[1], v[2]).astype(int))
            elif isinstance(v[0], float):
                exp[k] = list(np.around(np.linspace(v[0], v[1], v[2]), 4))
            else:
                assert False, "Experiments can only iterate over ints and floats."
        else:
            exp[k] = [v]

    runs = []
    for algo in exp['algo']:
        for data in exp['data']:
            for R in exp['R']:
                for C in exp['C']:
                    for K in exp['K']:
                        for E in exp['E']:
                            for B in exp['B']:
                                for eta in exp['eta']:
                                    runs.append({'algo': algo,
                                                 'data': data,
                                                  'R': R,
                                                  'C': C,
                                                  'K': K,
                                                  'E': E,
                                                  'B': B,
                                                  'eta': eta})
    return runs


def get_name(exp):
    return f"{exp['algo']}_{exp['data']}_R{exp['R']}_C{exp['C']}_K{exp['K']}_E{exp['E']}_B{exp['B']}_eta{exp['eta']}"


def run(dict):
    exp_name = get_name(dict)
    experiments = expand_experiment(dict)
    results = []
    for exp in experiments:
        print(exp)
        run_name = get_name(exp)
        history = {}
        history['experiment'] = exp_name
        history['run'] = run_name

        fm = Foreman(exp)
        history = {**history, **fm.run()} # merge dictionaries
        results.append(history)

    for r in results:
        print(r)
    return results


if __name__ == '__main__':
    import json

    str = '''{
        "algo": "FedAvg",
        "data": "MNIST_IID",
        "R": 2,
        "C": [0.3, 0.9, 3],
        "K": [2, 6, 3],
        "E": 10,
        "B": 64,
        "eta": 0.01 
        }'''
    print(get_name(json.loads(str)))
    for e in expand_experiment(json.loads(str)):
        print(get_name(e))
        print(e)