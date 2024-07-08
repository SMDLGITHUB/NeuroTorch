import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
import sys
import pandas as pd
import math
import copy
import Parameter
import time
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 200
sigma = 0

lr_init = 0.05
# decay_factor = 0.9
# step_size = 20
num_epoch = 200

class VGG9(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = HardwareConv(3, 64, 3, padding=1, bias = False)
        self.Conv2 = HardwareConv(64, 64, 3, padding=1, bias = False)
        self.Conv3 = HardwareConv(64, 128, 3, padding=1, bias = False)
        self.Conv4 = HardwareConv(128, 128, 3, padding=1, bias = False)
        self.Conv5 = HardwareConv(128, 256, 3, padding=1, bias = False)
        self.Conv6 = HardwareConv(256, 256, 3, padding=1, bias = False)
        self.Conv7 = HardwareConv(256, 256, 3, padding=1, bias = False)
        self.FC1 = HardwareLinear(256 * 4 * 4, 1024, bias = False)
        self.FC2 = HardwareLinear(1024, 10, bias = False)
        self.RELU = nn.ReLU()
        self.POOL = nn.AvgPool2d(2, 2)
        self.DROP1 = nn.Dropout2d(p = 0.1)
        self.DROP2 = nn.Dropout(p = 0.5)


    def forward(self, input):
        out = self.DROP1(self.RELU(self.Conv1(input)))
        out = self.RELU(self.Conv2(out))
        out = self.DROP1(self.POOL(out))
        out = self.DROP1(self.RELU(self.Conv3(out)))
        out = self.RELU(self.Conv4(out))
        out = self.POOL(out)
        out = self.DROP1(self.RELU(self.Conv5(out)))
        out = self.DROP1(self.RELU(self.Conv6(out)))
        out = self.RELU(self.Conv7(out))
        out = self.DROP1(self.POOL(out))
        out = out.view(-1, self.num_flat_features(out))
        out = self.DROP2(self.RELU(self.FC1(out)))
        out = self.FC2(out)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class HardwareConv(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(HardwareConv, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        Outp1 = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)

        Outp1 = quant_dev.apply(Outp1, 0, Parameter.ADCMax)

        Out = Outp1

        return Out


class HardwareLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(HardwareLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        Outpos = F.linear(input, self.weight)
        Outpos = quant_dev.apply(Outpos, 0, Parameter.ADCMax)

        Out = Outpos
        return Out


def quant_dev_weight(x, level):
    x_ = x.clone().to(device)
    #        index_separator = torch.linspace(0, torch.max(x_).item(), level).to(device)
    x_ = x_.clip(min=-1, max=1)
    index_separator = torch.linspace(-1, 1, 2**(level+1)+1).to(device)
    # n = torch.normal(0, 0.3, size=x_.shape).to(device) * index_separator[1]
    _, x_ = torch.min(torch.abs(x_.unsqueeze(-1) - index_separator), dim=-1)
    return index_separator[x_]

class quant_dev_update(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        x_ = x.clone()
        #        index_separator = torch.linspace(0, torch.max(x_).item(), level).to(device)
        x_ = x_.clip(min=-1, max=1)
        index_separator = torch.linspace(-1, 1, 2**11).to(device)
        # n = torch.normal(0, 0.3, size=x_.shape).to(device) * index_separator[1]
        x_ = index_separator[torch.searchsorted(index_separator, x_)]
        #        x_ = x_.clip(min=0, max=7)
        return x_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class quant_dev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, level, Max):
        ctx.set_materialize_grads(False)
        x_ = x.clone()
        #index_separator = torch.linspace(0, torch.max(x_).item(), level).to(device)
        #index_separator = torch.linspace(0, Max, level).to(device)
        # n = torch.normal(0, 0.3, size=x_.shape).to(device) * index_separator[1]
        #x_ = index_separator[torch.searchsorted(index_separator, x_ + index_separator[1] / 2) - 1]
        x_ = x_.clip(min=0, max=Max)
        return x_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def step_decay(optimizer, epoch, lr_init, decay_factor, step_size):
    lr = lr_init * math.pow(decay_factor, math.floor((1 + epoch) / step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def apply_readnoise(model, feature, onchip_sigma):
    with torch.no_grad():
        feature = feature * torch.normal(mean=1, std=onchip_sigma, size=model.weight.shape).clip(min=0).to(device)
        model.weight = copy.deepcopy(Parameter(feature))
    return

def ltpltd (gp0, gn0, delta, ltpNL, ltdNL) :
    delta_positive = delta.clamp(min=0)
    delta_negative = -delta.clamp(max=0)
    if ltpNL == 0:
        delta_positive_LTPupdated = 0.5 * delta_positive
        delta_negative_LTPupdated = 0.5 * delta_negative

    if ltpNL != 0:
        LTP_alpha = 1/(1-math.exp(-ltpNL))
        #delta_positive_LTPupdated : G+를 LTP 해줌
        delta_positive_LTPupdated = 0.5 * delta_positive*(ltpNL*(LTP_alpha-gp0))
        # delta_negative_LTPupdated : G-를 LTP 해줌 (이 결과는 양수임)
        delta_negative_LTPupdated = 0.5 * delta_negative * (ltpNL * (LTP_alpha - gn0))

    if ltdNL == 0:
        delta_positive_LTDupdated = -0.5 * delta_positive
        delta_negative_LTDupdated = -0.5 * delta_negative

    if ltdNL != 0:
        LTD_alpha = 1 / (1 - math.exp(-ltdNL))
        # delta_positive_LTPupdated : G-를 LTD 해줌 (이 결과는 음수임)
        delta_positive_LTDupdated = 0.5 * delta_positive * (-ltdNL * (LTD_alpha - 1 + gn0))
        # delta_negative_LTDupdated : G+를 LTD 해줌 (이 결과는 음수임, delta_negative가 양수라서)
        delta_negative_LTDupdated = 0.5 * delta_negative * (-ltdNL * (LTD_alpha - 1 + gp0))


    return (delta_positive_LTPupdated-delta_positive_LTDupdated), (delta_negative_LTPupdated - delta_negative_LTDupdated)




# Xavier
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)

transform = transforms.Compose([
        #transforms.RandAugment(),
        transforms.RandomChoice([transforms.AutoAugment(),transforms.RandAugment(),]),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=False, download=True, transform=transform2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=32)


loss_fn = nn.CrossEntropyLoss()
best_acc = 0
model_test = VGG9().to(device)
model_test.apply(init_weights)

model_test = model_test.to(device)



with torch.no_grad():
    Conv1_pos = torch.where(model_test.Conv1.weight > 0, model_test.Conv1.weight, model_test.Conv1.weight * 0)
    Conv1_neg = torch.where(model_test.Conv1.weight < 0, (-1) * model_test.Conv1.weight, model_test.Conv1.weight * 0)
    Conv2_pos = torch.where(model_test.Conv2.weight > 0, model_test.Conv2.weight, model_test.Conv2.weight * 0)
    Conv2_neg = torch.where(model_test.Conv2.weight < 0, (-1) * model_test.Conv2.weight, model_test.Conv2.weight * 0)
    Conv3_pos = torch.where(model_test.Conv3.weight > 0, model_test.Conv3.weight, model_test.Conv3.weight * 0)
    Conv3_neg = torch.where(model_test.Conv3.weight < 0, (-1) * model_test.Conv3.weight, model_test.Conv3.weight * 0)
    Conv4_pos = torch.where(model_test.Conv4.weight > 0, model_test.Conv4.weight, model_test.Conv4.weight * 0)
    Conv4_neg = torch.where(model_test.Conv4.weight < 0, (-1) * model_test.Conv4.weight, model_test.Conv4.weight * 0)
    Conv5_pos = torch.where(model_test.Conv5.weight > 0, model_test.Conv5.weight, model_test.Conv5.weight * 0)
    Conv5_neg = torch.where(model_test.Conv5.weight < 0, (-1) * model_test.Conv5.weight, model_test.Conv5.weight * 0)
    Conv6_pos = torch.where(model_test.Conv6.weight > 0, model_test.Conv6.weight, model_test.Conv6.weight * 0)
    Conv6_neg = torch.where(model_test.Conv6.weight < 0, (-1) * model_test.Conv6.weight, model_test.Conv6.weight * 0)
    Conv7_pos = torch.where(model_test.Conv7.weight > 0, model_test.Conv7.weight, model_test.Conv7.weight * 0)
    Conv7_neg = torch.where(model_test.Conv7.weight < 0, (-1) * model_test.Conv7.weight, model_test.Conv7.weight * 0)
    FC1_pos = torch.where(model_test.FC1.weight > 0, model_test.FC1.weight, model_test.FC1.weight * 0)
    FC1_neg = torch.where(model_test.FC1.weight < 0, (-1) * model_test.FC1.weight, model_test.FC1.weight * 0)
    FC2_pos = torch.where(model_test.FC2.weight > 0, model_test.FC2.weight, model_test.FC2.weight * 0)
    FC2_neg = torch.where(model_test.FC2.weight < 0, (-1) * model_test.FC2.weight, model_test.FC2.weight * 0)

for epoch in range(num_epoch):
    print(f"====== {epoch + 1} epoch of {num_epoch} ======")

    model_test.train()

    train_loss = 0
    valid_loss = 0
    correct = 0
    total_cnt = 0

    for step, batch in enumerate(train_loader):
        # Train Phase
        optimizer = optim.SGD(model_test.parameters(), lr=lr_init)
        #optimizer = step_decay(optimizer, epoch, lr_init, decay_factor, step_size)
        batch[0], batch[1] = batch[0].to(device), batch[1].to(device)

        logits = model_test(batch[0])
        loss = loss_fn(logits, batch[1])
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        class_init = 0
        train_loss += loss.item()
        _, predict = logits.max(1)

        total_cnt += batch[1].size(0)
        correct += predict.eq(batch[1]).sum().item()

        if step % 500 == 0 and step != 0:
            print(f"\n====== {step} Step of {len(train_loader)} ======")
            print(f"Train Acc : {correct / total_cnt}")
            print(f"Train Loss : {loss.item() / batch[1].size(0)}")

    correct = 0
    total_cnt = 0

    # Test Phase
    with torch.no_grad():
        model_test.eval()
        for step, batch in enumerate(test_loader):

            # input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model_test(batch[0])
            valid_loss += loss_fn(logits, batch[1])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
        valid_acc = correct / total_cnt
        print(f"\nValid Acc : {valid_acc}")
        print(f"Valid Loss : {valid_loss / total_cnt}")


        if (valid_acc > best_acc):
            best_acc = valid_acc
            torch.save(model_test.state_dict(), 'NeuroTorch_VGG9_240601_Pretrain' + str(sigma))
            print("Model Saved!")

# sys.stdout.close()