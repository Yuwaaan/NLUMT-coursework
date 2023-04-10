import torch
import numpy

# model=torch.load("question4/checkpoint_last.pt")
# print(model.keys())
# # print(model['model'])
# # print(model['val_loss'])
# # print(model)
# for key in model['model']:
#     print(key)
#     # print(model['model'][key])



import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

# train_loss_baseline = []
# val_loss_baseline = []
# count_baseline = 0
#
# file1 = open("baseline_jilu.txt")
# for line in file1:
#     line = line.split()
#     print(line)
#     if count_baseline % 2 == 0:
#         train_l = float(line[4])
#         train_loss_baseline.append(train_l)
#         count_baseline += 1
#     else:
#         val_l = float(line[4])
#         val_loss_baseline.append(val_l)
#         count_baseline += 1

# train_loss = []
# val_loss = []
# count = 0
#
# file = open("question4_jilu.txt")
# for line in file:
#     line = line.split()
#     print(line)
#     if count % 2 == 0:
#         train_l = float(line[3])
#         train_loss.append(train_l)
#         count += 1
#     else:
#         val_l = float(line[3])
#         val_loss.append(val_l)
#         count += 1
# print(len(train_loss))
# print(train_loss)
# print(len(val_loss))
# print(val_loss)


# plt.clf()
# x_grid = np.arange(0, 100, 1)
# plt.plot(x_grid, val_loss_baseline, label='Baseline valid loss')
# plt.plot(x_grid, train_loss_baseline, label='Baseline training loss')
# plt.plot(x_grid, val_loss, label='2encoder-3decoder valid loss')
# plt.plot(x_grid, train_loss, label='2encoder-3decoder training loss')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

train_loss_baseline = []
val_loss_baseline = []
count_baseline = 0

file1 = open("question7_jilu.txt")
for line in file1:
    line = line.split()
    print(line)
    if count_baseline % 2 == 0:
        train_l = float(line[4])
        train_loss_baseline.append(train_l)
        count_baseline += 1
    else:
        val_l = float(line[4])
        val_loss_baseline.append(val_l)
        count_baseline += 1

plt.clf()
x_grid = np.arange(0, len(train_loss_baseline), 1)
plt.plot(x_grid, val_loss_baseline, label='Transformer valid loss')
plt.plot(x_grid, train_loss_baseline, label='Transformer training loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()