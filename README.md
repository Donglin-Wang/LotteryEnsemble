# LotteryEnsemble: LotteryFL re-implementation and experiments

This repository is used for experimenting with the LotteryFL model. The original paper can be found [here](https://arxiv.org/abs/2008.03371). As of right now, we are trying to combine the functionalities of these three repositories. Some of the code will be direct copies from the repositories listed below, with minor changes to the variable names and structure.
1. https://github.com/AshwinRJ/Federated-Learning-PyTorch Implementation of the FedAvg algorithm. The original paper for FedAvg can be found [here](https://arxiv.org/abs/1602.05629)
2. https://github.com/jeremy313/non-iid-dataset-for-personalized-federated-learning Offical repo for LotteryFL. As of 2020/10/14, this repo only contains the code for generating their dataset.
3. https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch Re-implementation of the Lottery Ticket Hypothesis (LTH). The original paper for LTH can be found [here](https://arxiv.org/abs/1803.03635)

## Preliminary Goals:

- See how much overlap there is for different weights coming from different clients
- See how weighted average affect the performance of LotteryFL

## Preliminary Plan:

- **2020/10/14:** Create repo, decide on a bare-bone project structure
- Starting **2020/10/15:** Each contributor come up with a pseudo-code for the Client & Server update logic
- By **2020/10/16** Finalize on the client and server update logic
- By **2020/10/19** Implement a functional single-round LTH experiment in the Client class

## Potential Adaptations:

- [Ensemble Distillation for Robust Model Fusion in Federated Learning](https://arxiv.org/abs/2006.07242)
- [Sparse Transfer Learning via Winning Lottery Tickets](https://arxiv.org/abs/1905.07785)
	- Implementation: https://github.com/rahulsmehta/sparsity-experiments
- [Model-Agnostic Round-Optimal Federated Learning via Knowledge Transfer](https://arxiv.org/abs/2010.01017)
- [Using Winning Lottery Tickets in Transfer Learning for Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8852405) (Please access through Uottawa bibilio or other platform with subscription to IEEE Xplore)
- [FedMD: Heterogenous Federated Learning via Model Distillation](https://arxiv.org/pdf/1910.03581v1.pdf)
	- Implementation: https://github.com/diogenes0319/FedMD_clean
- [One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://arxiv.org/pdf/1906.02773v2.pdf)
    - Implementation: https://github.com/varungohil/Generalizing-Lottery-Tickets (**Non-official**)
- [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671) (**Paper by the original LTH author**)

## Details:

### Global Arguments:

Here are the configs shared across the program in the `args` variable. To access individual config, use `args.<option name>`. For example, `args.prune_type`

**General Configs**

- `--dataset`	: Choice of dataset 
	- Options : `mnist`
	- Default : `mnist`
- `--arch`	 : Type of architecture
- `--gpu`	: Decide Which GPU the program should use 

**Configs for LTH**

- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 

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