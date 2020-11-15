def expand_experiment(dict):
    exp = {}
    for k, v in dict.items():
        expand = False
        if isinstance(v, list):
            exp[k] = list(range(v[0], v[1], v[2]))
            print(exp[k])
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
                                                 'data': eta,
                                                  'R': R,
                                                  'C': C,
                                                  'K': K,
                                                  'E': E,
                                                  'B': B,
                                                  'eta': eta})
    return runs

if __name__ == '__main__':
    import json

    str = '''{
        "algo": "FedAvg",
        "data": "MNIST_IID",
        "R": 2,
        "C": 0.5,
        "K": [2, 7, 2],
        "E": [4, 42, 37],
        "B": 64,
        "eta": 0.01 
        }'''
    for e in expand_experiment(json.loads(str)):
        print(e)