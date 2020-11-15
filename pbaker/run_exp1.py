from experiment import run

exp = {
    "algo": "FedAvg",
    "data": "MNIST_IID",
    "R": 2,
    "C": [0.3, 0.9, 1],
    "K": [20, 100, 5],
    "E": 10,
    "B": 64,
    "eta": 0.01
}
run(exp)
