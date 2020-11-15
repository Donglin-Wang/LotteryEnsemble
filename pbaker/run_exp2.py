from experiment import run

exp = {
    "algo": "LotteryFL",
    "data": "MNIST_IID",
    "R": 2,
    "C": 0.4,
    "K": [5, 40, 8],
    "E": 10,
    "B": 64,
    "eta": 0.01,
    'acc_threshold': .5,
    'r_target': .25,
    'r_p': .15
}
run(exp)
