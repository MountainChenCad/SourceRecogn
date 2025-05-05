# 学习率参数，使学习率随着迭代次数增多而下降
def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer


schedule_dict = {"inv":inv_lr_scheduler}


'''
schedule_dict = {"inv":inv_lr_scheduler}
lr=0.001
gamma=0.001
power=0.75
for i in range(12001):
    lr = 0.01 * (1 + 10 * i/12000) ** (-power)
    if i <200:
        print(lr)

from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch
import network, my_loss
base_network = network.CNN(8)
parameter_list = base_network.get_parameters() 
optimizer_config = {"type":torch.optim.Adam, "optim_params":
                    {'lr':0.002}}
optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))
scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)
for i in range(12000):
    scheduler.step()
    cur_lr=optimizer.param_groups[-1]['lr']
    print('cur_lr:',cur_lr)
'''
