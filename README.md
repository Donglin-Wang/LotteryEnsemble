# LotteryEnsemble: LotteryFL re-implementation and experiments

This repository is used for experimenting with the LotteryFL model. The original paper can be found [here](https://arxiv.org/abs/2008.03371). As of right now, we are trying to combine the functionalities of these three repositories:
1. https://github.com/AshwinRJ/Federated-Learning-PyTorch Implementation of the FedAvg algorithm. The original paper for FedAvg can be found [here](https://arxiv.org/abs/1602.05629)
2. https://github.com/jeremy313/non-iid-dataset-for-personalized-federated-learning Offical repo for LotteryFL. As of 2020/10/14, this repo only contains the code for generating their dataset.
3. https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch Re-implementation of the Lottery Ticket Hypothesis (LTH). The original paper for LTH can be found [here](https://arxiv.org/abs/1803.03635)

## Preliminary Plan:
- **2020/10/14:** Create repo, decide on a bare-bone project structure

## Details:

Here are the configs shared across the program in the `args` variable. To access individual config, use `args.<option name>`. For example, `args.prune_type`

### Global Arguments:

**General Configs**

- `--dataset`	: Choice of dataset 
	- Options : `mnist`
	- Default : `mnist`
- `--arch`	 : Type of architecture
- `--gpu`	: Decide Which GPU the program should use 

**Configs for LTH**

- `--prune_type` : Type of pruning 
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
- `--prune_iterations`	: Number of cycle of pruning that should be done. 

- `--batch_size`	: Batch size.
	- Default : `4`
- `--log_freq`	: Frequency for printing accuracy and loss. 
- `--valid_freq`	: Frequency for Validation.

**Configs for FedAvg**

- `--server_epoch`   : Number of communication rounds for the server.
- `--client_epoch`   : Number of rounds of training for each client.
- `--lr`       : Learning rate
- `--iid`      : Distribution of data amongst users. 
- `--num_users`: Number of users. 
- `--frac`     : Fraction of users to be used for federated updates. 
- `--local_ep` : Number of local training epochs in each user. 
- `--local_bs` : Batch size of local updates in each user. 
- `--unequal`  : Used in non-iid setting. Option to split the data amongst users equally or unequally.
