# 测试神经网络模型
import torch
from torch.utils.tensorboard import SummaryWriter

import data_loader

writer = SummaryWriter('logs')
# 读取模型
recognition = torch.load('pth/test_19.pth')  # 已有模型路径
print(recognition)

# 模型预测的数字数量
num_find = [0] * 10
# 正确的数字数量
num_true = [0] * 10
# 每个数字正确的概率
num_accuracy = [0.0] * 10
# 总正确概率
total_accuracy = 0.0

with torch.no_grad():
    for data in data_loader.test_dataloader:
        labels, images = data
        images = images.unsqueeze(1)
        if torch.cuda.is_available():
            labels = labels.cuda()
            images = images.cuda()
        output = recognition(images)
        num_prediect = output.argmax(1)
        for i in range(len(num_prediect)):
            # 记录预测数字
            num_find[int(num_prediect[i])] += 1
            # 记录真实数字
            num_true[int(labels[i])] += 1
            # 记录正确预测
            if num_find[int(num_prediect[i])] == num_true[int(labels[i])]:
                num_accuracy[int(num_prediect[i])] += 1
        accuracy = (num_prediect == labels).sum()
        total_accuracy += accuracy

for i in range(len(num_accuracy)):
    num_accuracy[i] = (1 - abs(num_true[i] - num_find[i]) / num_true[i]) * 100
    writer.add_scalar('预测数量', num_find[i], i)
    writer.add_scalar('每数正确率', num_accuracy[i], i)

print(f'测试集正确率:{100 * total_accuracy / data_loader.test_data_size:.2f}%')
print(f'预测：{num_find}')
print(f'真实：{num_true}')
print(f'概率:{num_accuracy}')
writer.close()
