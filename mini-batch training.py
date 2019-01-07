# -*- coding:utf-8 -*-
__author__ = 'GIS'
import torch
import torch.utils.data as Data

# torch.manual_seed(1)

BATCH_SIZE = 7
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)  # 训练为x，目标y
loader = Data.DataLoader(  # 将训练变成一 批次
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 要不要打乱选取样本的顺序
    # num_workers=2,  # 2个进程提取数据
)

print(type(loader))
print(loader)

for epoch in range(3):  # 训练3次
    for step,(batch_x, batch_y) in enumerate(loader):  # enumerate 提取的时候施加一个索引
        # training.....
        print('Epoch:', epoch, '\nstep:', step, '\nbatch_x:', batch_x.numpy(),
              '\nbatch_y:', batch_y.numpy())
