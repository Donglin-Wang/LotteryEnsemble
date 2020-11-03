# This is just to make it easy for everyone to run the code for now. This will be replaced by
# a JSON file-based setup.

from foreman import Foreman


fm = Foreman({
    'algo': 'LotteryFL',
    'data': 'MNIST_IID',
    'R': 10,
    'C': 1,
    'K': 3,
    'E': 4,
    'B': 64,
    'eta': 0.01, # currently unused
    'acc_threshold': .5,
    'r_target': .25,
    'r_p': .25
    })
fm.run()