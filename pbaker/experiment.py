import os
import numpy as np
import pandas as pd
from foreman import Foreman

DATA_FOLDER     = '../pbaker_experiments'
DF_EXPS_FNAME   = os.path.join(DATA_FOLDER, 'df_experiments.pkl')
DF_RUNS_FNAME   = os.path.join(DATA_FOLDER, 'df_runs.pkl')
DF_EXPS         = None
DF_RUNS         = None
ALL_HYPERS      = ['algo', 'data', 'R', 'C', 'K', 'E', 'B', 'eta', 'acc_threshold', 'r_target', 'r_p']
DF_EXPS_COLUMNS = ['experiment', 'run']
DF_RUNS_COLUMNS = ['run',] + ALL_HYPERS + \
                  ['server_loss_initial', 'server_acc_initial', 'server_loss_final', 'server_acc_final',
                   'client_loss_final', 'client_acc_final']


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
                                    for acc_threshold in exp.get('acc_threshold', [0.0]):
                                        for r_target in exp.get('r_target', [0.0]):
                                            for r_p in exp.get('r_p', [0.0]):
                                                runs.append({'algo': algo,
                                                             'data': data,
                                                             'R': R,
                                                             'C': C,
                                                             'K': K,
                                                             'E': E,
                                                             'B': B,
                                                             'eta': eta,
                                                             'acc_threshold': acc_threshold,
                                                             'r_target': r_target,
                                                             'r_p': r_p})
    return runs


def get_name(exp):
    return f"{exp['algo']}_{exp['data']}_R{exp['R']}_C{exp['C']}_K{exp['K']}_E{exp['E']}_B{exp['B']}_eta{exp['eta']}"\
           f"_Acc{exp.get('acc_threshold', 0.0)}_rtrg{exp.get('r_target', 0.0)}_rprn{exp.get('r_p', 0.0)}"


def load_dataframes():
    global DF_EXPS, DF_RUNS
    if not DF_EXPS:
        print(DF_EXPS_FNAME)
        if not os.path.isfile(DF_EXPS_FNAME):
            if not os.path.exists(DATA_FOLDER):
                os.makedirs(DATA_FOLDER)
            print("Creating DataFrame files in: ", DATA_FOLDER)
            pd.DataFrame(columns=DF_EXPS_COLUMNS).to_pickle(DF_EXPS_FNAME)
            pd.DataFrame(columns=DF_RUNS_COLUMNS).to_pickle(DF_RUNS_FNAME)
    DF_RUNS = pd.read_pickle(DF_RUNS_FNAME)
    DF_EXPS = pd.read_pickle(DF_EXPS_FNAME)

def save_dataframes():
    DF_EXPS.to_pickle(DF_EXPS_FNAME)
    DF_RUNS.to_pickle(DF_RUNS_FNAME)


def run(dict):
    global DF_EXPS, DF_RUNS

    load_dataframes()

    exp_name = get_name(dict)
    experiments = expand_experiment(dict)

    results = []
    for exp in experiments:
        print(exp)
        run_name = get_name(exp)

        if DF_EXPS[(DF_EXPS['experiment'] == exp_name) & (DF_EXPS['run'] == run_name)].empty:
            DF_EXPS = DF_EXPS.append({'experiment': exp_name, 'run': run_name}, ignore_index=True)

        if not run_name in DF_RUNS['run'].values:
            history = {}
            history['run'] = run_name

            fm = Foreman(exp)
            history = {**history, **exp, **fm.run()}  # merge dictionaries
            DF_RUNS = DF_RUNS.append(history, ignore_index=True)
        else:
            history = DF_RUNS.loc[DF_RUNS['run'] == run_name].iloc[0].to_dict()
        results.append(history)

    print('DONE --------------------------')
    for r in results:
        print(r)

    print(DF_RUNS.shape)
    save_dataframes()
    return results


if __name__ == '__main__':
    load_dataframes()
    print(DF_EXPS)
    print(DF_RUNS)
    assert False
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