from experiment import run

exp = {
    "algo": "FedAvg",
    "data": "MNIST_IID",
    "R": 2,
    "C": [0.3, 0.9, 3],
    "K": [2, 6, 3],
    "E": 10,
    "B": 64,
    "eta": 0.01
}
run(exp)
