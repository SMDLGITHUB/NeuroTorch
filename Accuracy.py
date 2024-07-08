import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
import sys
import math
import Parameter

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ReadSynEnergy = 0
UpdateCurrEnergy = 0
UpdateCapEnergy = 0

def Calculate():
    global UpdateCurrEnergy
    global UpdateCapEnergy

    # Check Electro Migration Limitation Here
    if Parameter.Mode == 'Onchip_Parallel':
        print('ELECTROMIGRATION CHECK!!')

        MAXWRICURR = Parameter.WriteCurr * Parameter.SubCol     # Worst Case of Parallel Outer Product Update (All NVMs are Write Operation)
        NORMALWRICURR = Parameter.WriteCurr * Parameter.SubCol * 0.25     # Assume only 10% of devices are updated once (User might modify here) (We have checked it roughly)
        NORMALWRICURR += Parameter.InhibCurr * Parameter.SubCol * (1 - 0.25)     # Rest of the NVM cells are inhibited but current might flow even if inhibited (If weight is 1)
        NORMALWRICURR *= 0.14       # Average Conductance in the largest Layer (Conv1 (We have Checked it in VGG-9) User should modify it to their own network.)
        MAXINHIBCURR = Parameter.InhibCurr * Parameter.SubCol   # Worst Case of Parallel Outer Product Update (All NVMs are Inhibited)
        MININHIBCURR = Parameter.InhibCurr * Parameter.SubCol / Parameter.OnOffRatio    # Best Case of Parallel Outer Product Update (All NVMs are Inhibited in Off State)

        if MAXWRICURR >= Parameter.MIGRATIONLIM:    # No Error But should Care About the Warning
            print('Beware that the worst case write current is over the current limit!!')

        if MAXINHIBCURR >= Parameter.MIGRATIONLIM:    # No Error But should Care About the Warning
            print('Beware that the worst case inhibited current is over the current limit!!')

        if NORMALWRICURR >= Parameter.MIGRATIONLIM:     # Error (Do not Meet EM condition)
            print('CURRENT LIMIT IS VIOLATED WHEN PARALLEL UPDATE!! (In Typical Parallel Weight Update!!)')
            sys.exit()

        if MININHIBCURR >= Parameter.MIGRATIONLIM:      # Error (Do not Meet EM condition)
            print('CURRENT LIMIT IS VIOLATED WHEN PARALLEL UPDATE!! (EVEN IN THE BEST CASE!!)')
            sys.exit()

        print('CURRENT LIMIT IS MEETED!!')      # But should Care about the warning for potential EM problem.

    BATCH_SIZE = Parameter.BATCH
    lr_init = Parameter.LearnRate  # 0.0005
    num_epoch = Parameter.EPOCH
    sigma = Parameter.Vari
    level = Parameter.LEVEL
    Min1 = 1/(Parameter.OnOffRatio * Parameter.Kernel * Parameter.Kernel)   # Normalized Gmin for Conv Layers
    Min2 = 1/(Parameter.OnOffRatio * (4096/ Parameter.SubRow))   # Normalized Gmin for FC1 Layers
    Min3 = 1/(Parameter.OnOffRatio * (1024/ Parameter.SubRow))   # Normalized Gmin for FC2 Layers

    loss_fn = nn.CrossEntropyLoss()
    model_test = VGG9().to(device)
    model_test.apply(init_weights)
    model_test = model_test.to(device)

    transform = transforms.Compose([
        # transforms.RandAugment(),
        # transforms.RandomChoice([transforms.AutoAugment(),transforms.RandAugment(),]),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=False, download=True,
                                                transform=transform2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=16)

    if Parameter.Mode == 'Inference_Normal':

        model_test.load_state_dict(torch.load('NeuroTorch_VGG9_240531_Pretrain0'))

        with torch.no_grad():
            # Conductance is always positive
            Conv1_pos = torch.where(model_test.Conv1.weight > 0, model_test.Conv1.weight, model_test.Conv1.weight * 0)
            Conv1_neg = torch.where(model_test.Conv1.weight < 0, (-1) * model_test.Conv1.weight,
                                    model_test.Conv1.weight * 0)
            Conv2_pos = torch.where(model_test.Conv2.weight > 0, model_test.Conv2.weight, model_test.Conv2.weight * 0)
            Conv2_neg = torch.where(model_test.Conv2.weight < 0, (-1) * model_test.Conv2.weight,
                                    model_test.Conv2.weight * 0)
            Conv3_pos = torch.where(model_test.Conv3.weight > 0, model_test.Conv3.weight, model_test.Conv3.weight * 0)
            Conv3_neg = torch.where(model_test.Conv3.weight < 0, (-1) * model_test.Conv3.weight,
                                    model_test.Conv3.weight * 0)
            Conv4_pos = torch.where(model_test.Conv4.weight > 0, model_test.Conv4.weight, model_test.Conv4.weight * 0)
            Conv4_neg = torch.where(model_test.Conv4.weight < 0, (-1) * model_test.Conv4.weight,
                                    model_test.Conv4.weight * 0)
            Conv5_pos = torch.where(model_test.Conv5.weight > 0, model_test.Conv5.weight, model_test.Conv5.weight * 0)
            Conv5_neg = torch.where(model_test.Conv5.weight < 0, (-1) * model_test.Conv5.weight,
                                    model_test.Conv5.weight * 0)
            Conv6_pos = torch.where(model_test.Conv6.weight > 0, model_test.Conv6.weight, model_test.Conv6.weight * 0)
            Conv6_neg = torch.where(model_test.Conv6.weight < 0, (-1) * model_test.Conv6.weight,
                                    model_test.Conv6.weight * 0)
            Conv7_pos = torch.where(model_test.Conv7.weight > 0, model_test.Conv7.weight, model_test.Conv7.weight * 0)
            Conv7_neg = torch.where(model_test.Conv7.weight < 0, (-1) * model_test.Conv7.weight,
                                    model_test.Conv7.weight * 0)
            FC1_pos = torch.where(model_test.FC1.weight > 0, model_test.FC1.weight, model_test.FC1.weight * 0)
            FC1_neg = torch.where(model_test.FC1.weight < 0, (-1) * model_test.FC1.weight, model_test.FC1.weight * 0)
            FC2_pos = torch.where(model_test.FC2.weight > 0, model_test.FC2.weight * 0.9, model_test.FC2.weight * 0)
            FC2_neg = torch.where(model_test.FC2.weight < 0, (-1) * model_test.FC2.weight * 0.9,
                                  model_test.FC2.weight * 0)

            # Weights are from Gmin(Min) to Gmax(1)
            Conv1_pos = Conv1_pos.clip(min=Min1, max=1)
            Conv1_neg = Conv1_neg.clip(min=Min1, max=1)
            Conv2_pos = Conv2_pos.clip(min=Min1, max=1)
            Conv2_neg = Conv2_neg.clip(min=Min1, max=1)
            Conv3_pos = Conv3_pos.clip(min=Min1, max=1)
            Conv3_neg = Conv3_neg.clip(min=Min1, max=1)
            Conv4_pos = Conv4_pos.clip(min=Min1, max=1)
            Conv4_neg = Conv4_neg.clip(min=Min1, max=1)
            Conv5_pos = Conv5_pos.clip(min=Min1, max=1)
            Conv5_neg = Conv5_neg.clip(min=Min1, max=1)
            Conv6_pos = Conv6_pos.clip(min=Min1, max=1)
            Conv6_neg = Conv6_neg.clip(min=Min1, max=1)
            Conv7_pos = Conv7_pos.clip(min=Min1, max=1)
            Conv7_neg = Conv7_neg.clip(min=Min1, max=1)
            FC1_pos = FC1_pos.clip(min=Min2, max=1)
            FC1_neg = FC1_neg.clip(min=Min2, max=1)
            FC2_pos = FC2_pos.clip(min=Min3, max=1)
            FC2_neg = FC2_neg.clip(min=Min3, max=1)

            # Conductance is quantized according to the predefined levels
            Conv1_pos = quant_dev_weight(Conv1_pos, level)
            Conv1_neg = quant_dev_weight(Conv1_neg, level)
            Conv2_pos = quant_dev_weight(Conv2_pos, level)
            Conv2_neg = quant_dev_weight(Conv2_neg, level)
            Conv3_pos = quant_dev_weight(Conv3_pos, level)
            Conv3_neg = quant_dev_weight(Conv3_neg, level)
            Conv4_pos = quant_dev_weight(Conv4_pos, level)
            Conv4_neg = quant_dev_weight(Conv4_neg, level)
            Conv5_pos = quant_dev_weight(Conv5_pos, level)
            Conv5_neg = quant_dev_weight(Conv5_neg, level)
            Conv6_pos = quant_dev_weight(Conv6_pos, level)
            Conv6_neg = quant_dev_weight(Conv6_neg, level)
            Conv7_pos = quant_dev_weight(Conv7_pos, level)
            Conv7_neg = quant_dev_weight(Conv7_neg, level)
            FC1_pos = quant_dev_weight(FC1_pos, level)
            FC1_neg = quant_dev_weight(FC1_neg, level)
            FC2_pos = quant_dev_weight(FC2_pos, level)
            FC2_neg = quant_dev_weight(FC2_neg, level)

            # Conductance variation is multiplied after weight quantization (D-to-D variation)
            Conv1_pos *= torch.clip(torch.normal(1, sigma, size=Conv1_pos.size()).to(device), 0, 10)
            Conv1_neg *= torch.clip(torch.normal(1, sigma, size=Conv1_neg.size()).to(device), 0, 10)
            Conv2_pos *= torch.clip(torch.normal(1, sigma, size=Conv2_pos.size()).to(device), 0, 10)
            Conv2_neg *= torch.clip(torch.normal(1, sigma, size=Conv2_neg.size()).to(device), 0, 10)
            Conv3_pos *= torch.clip(torch.normal(1, sigma, size=Conv3_pos.size()).to(device), 0, 10)
            Conv3_neg *= torch.clip(torch.normal(1, sigma, size=Conv3_neg.size()).to(device), 0, 10)
            Conv4_pos *= torch.clip(torch.normal(1, sigma, size=Conv4_pos.size()).to(device), 0, 10)
            Conv4_neg *= torch.clip(torch.normal(1, sigma, size=Conv4_neg.size()).to(device), 0, 10)
            Conv5_pos *= torch.clip(torch.normal(1, sigma, size=Conv5_pos.size()).to(device), 0, 10)
            Conv5_neg *= torch.clip(torch.normal(1, sigma, size=Conv5_neg.size()).to(device), 0, 10)
            Conv6_pos *= torch.clip(torch.normal(1, sigma, size=Conv6_pos.size()).to(device), 0, 10)
            Conv6_neg *= torch.clip(torch.normal(1, sigma, size=Conv6_neg.size()).to(device), 0, 10)
            Conv7_pos *= torch.clip(torch.normal(1, sigma, size=Conv7_pos.size()).to(device), 0, 10)
            Conv7_neg *= torch.clip(torch.normal(1, sigma, size=Conv7_neg.size()).to(device), 0, 10)
            FC1_pos *= torch.clip(torch.normal(1, sigma, size=FC1_pos.size()).to(device), 0, 10)
            FC1_neg *= torch.clip(torch.normal(1, sigma, size=FC1_neg.size()).to(device), 0, 10)
            FC2_pos *= torch.clip(torch.normal(1, sigma, size=FC2_pos.size()).to(device), 0, 10)
            FC2_neg *= torch.clip(torch.normal(1, sigma, size=FC2_neg.size()).to(device), 0, 10)

            # weight is calculated through (G+ - G-)
            model_test.Conv1.weight = nn.Parameter(Conv1_pos - Conv1_neg)
            model_test.Conv2.weight = nn.Parameter(Conv2_pos - Conv2_neg)
            model_test.Conv3.weight = nn.Parameter(Conv3_pos - Conv3_neg)
            model_test.Conv4.weight = nn.Parameter(Conv4_pos - Conv4_neg)
            model_test.Conv5.weight = nn.Parameter(Conv5_pos - Conv5_neg)
            model_test.Conv6.weight = nn.Parameter(Conv6_pos - Conv6_neg)
            model_test.Conv7.weight = nn.Parameter(Conv7_pos - Conv7_neg)
            model_test.FC1.weight = nn.Parameter(FC1_pos - FC1_neg)
            model_test.FC2.weight = nn.Parameter(FC2_pos - FC2_neg)

        correct = 0
        total_cnt = 0
        valid_loss = 0

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
            print(f"\nValid Acc of Inferencing : {valid_acc}")
            print(f"Valid Loss of Inferencing : {valid_loss / total_cnt}")

    if Parameter.Mode == 'Onchip_Normal':
        print('lets go')

    if Parameter.Mode == 'Onchip_Parallel':

        model_test.load_state_dict(torch.load('NeuroTorch_VGG9_240531_Pretrain0'))

        with torch.no_grad():
            # Conductance is always positive
            Conv1_pos = torch.where(model_test.Conv1.weight > 0, model_test.Conv1.weight, model_test.Conv1.weight * 0)
            Conv1_neg = torch.where(model_test.Conv1.weight < 0, (-1) * model_test.Conv1.weight,
                                    model_test.Conv1.weight * 0)
            Conv2_pos = torch.where(model_test.Conv2.weight > 0, model_test.Conv2.weight, model_test.Conv2.weight * 0)
            Conv2_neg = torch.where(model_test.Conv2.weight < 0, (-1) * model_test.Conv2.weight,
                                    model_test.Conv2.weight * 0)
            Conv3_pos = torch.where(model_test.Conv3.weight > 0, model_test.Conv3.weight, model_test.Conv3.weight * 0)
            Conv3_neg = torch.where(model_test.Conv3.weight < 0, (-1) * model_test.Conv3.weight,
                                    model_test.Conv3.weight * 0)
            Conv4_pos = torch.where(model_test.Conv4.weight > 0, model_test.Conv4.weight, model_test.Conv4.weight * 0)
            Conv4_neg = torch.where(model_test.Conv4.weight < 0, (-1) * model_test.Conv4.weight,
                                    model_test.Conv4.weight * 0)
            Conv5_pos = torch.where(model_test.Conv5.weight > 0, model_test.Conv5.weight, model_test.Conv5.weight * 0)
            Conv5_neg = torch.where(model_test.Conv5.weight < 0, (-1) * model_test.Conv5.weight,
                                    model_test.Conv5.weight * 0)
            Conv6_pos = torch.where(model_test.Conv6.weight > 0, model_test.Conv6.weight, model_test.Conv6.weight * 0)
            Conv6_neg = torch.where(model_test.Conv6.weight < 0, (-1) * model_test.Conv6.weight,
                                    model_test.Conv6.weight * 0)
            Conv7_pos = torch.where(model_test.Conv7.weight > 0, model_test.Conv7.weight, model_test.Conv7.weight * 0)
            Conv7_neg = torch.where(model_test.Conv7.weight < 0, (-1) * model_test.Conv7.weight,
                                    model_test.Conv7.weight * 0)
            FC1_pos = torch.where(model_test.FC1.weight > 0, model_test.FC1.weight, model_test.FC1.weight * 0)
            FC1_neg = torch.where(model_test.FC1.weight < 0, (-1) * model_test.FC1.weight, model_test.FC1.weight * 0)
            FC2_pos = torch.where(model_test.FC2.weight > 0, model_test.FC2.weight * 0.9, model_test.FC2.weight * 0)
            FC2_neg = torch.where(model_test.FC2.weight < 0, (-1) * model_test.FC2.weight * 0.9,
                                  model_test.FC2.weight * 0)

            # Weights are from Gmin(Min) to Gmax(1)
            Conv1_pos = Conv1_pos.clip(min=Min1, max=1)
            Conv1_neg = Conv1_neg.clip(min=Min1, max=1)
            Conv2_pos = Conv2_pos.clip(min=Min1, max=1)
            Conv2_neg = Conv2_neg.clip(min=Min1, max=1)
            Conv3_pos = Conv3_pos.clip(min=Min1, max=1)
            Conv3_neg = Conv3_neg.clip(min=Min1, max=1)
            Conv4_pos = Conv4_pos.clip(min=Min1, max=1)
            Conv4_neg = Conv4_neg.clip(min=Min1, max=1)
            Conv5_pos = Conv5_pos.clip(min=Min1, max=1)
            Conv5_neg = Conv5_neg.clip(min=Min1, max=1)
            Conv6_pos = Conv6_pos.clip(min=Min1, max=1)
            Conv6_neg = Conv6_neg.clip(min=Min1, max=1)
            Conv7_pos = Conv7_pos.clip(min=Min1, max=1)
            Conv7_neg = Conv7_neg.clip(min=Min1, max=1)
            FC1_pos = FC1_pos.clip(min=Min2, max=1)
            FC1_neg = FC1_neg.clip(min=Min2, max=1)
            FC2_pos = FC2_pos.clip(min=Min3, max=1)
            FC2_neg = FC2_neg.clip(min=Min3, max=1)

            # Conductance is quantized
            Conv1_pos = quant_dev_weight(Conv1_pos, level)
            Conv1_neg = quant_dev_weight(Conv1_neg, level)
            Conv2_pos = quant_dev_weight(Conv2_pos, level)
            Conv2_neg = quant_dev_weight(Conv2_neg, level)
            Conv3_pos = quant_dev_weight(Conv3_pos, level)
            Conv3_neg = quant_dev_weight(Conv3_neg, level)
            Conv4_pos = quant_dev_weight(Conv4_pos, level)
            Conv4_neg = quant_dev_weight(Conv4_neg, level)
            Conv5_pos = quant_dev_weight(Conv5_pos, level)
            Conv5_neg = quant_dev_weight(Conv5_neg, level)
            Conv6_pos = quant_dev_weight(Conv6_pos, level)
            Conv6_neg = quant_dev_weight(Conv6_neg, level)
            Conv7_pos = quant_dev_weight(Conv7_pos, level)
            Conv7_neg = quant_dev_weight(Conv7_neg, level)
            FC1_pos = quant_dev_weight(FC1_pos, level)
            FC1_neg = quant_dev_weight(FC1_neg, level)
            FC2_pos = quant_dev_weight(FC2_pos, level)
            FC2_neg = quant_dev_weight(FC2_neg, level)

            # Conductance variation is multiplied after weight quantization (D-to-D variation)
            Conv1_pos *= torch.clip(torch.normal(1, sigma, size=Conv1_pos.size()).to(device), 0, 100)
            Conv1_neg *= torch.clip(torch.normal(1, sigma, size=Conv1_neg.size()).to(device), 0, 100)
            Conv2_pos *= torch.clip(torch.normal(1, sigma, size=Conv2_pos.size()).to(device), 0, 100)
            Conv2_neg *= torch.clip(torch.normal(1, sigma, size=Conv2_neg.size()).to(device), 0, 100)
            Conv3_pos *= torch.clip(torch.normal(1, sigma, size=Conv3_pos.size()).to(device), 0, 100)
            Conv3_neg *= torch.clip(torch.normal(1, sigma, size=Conv3_neg.size()).to(device), 0, 100)
            Conv4_pos *= torch.clip(torch.normal(1, sigma, size=Conv4_pos.size()).to(device), 0, 100)
            Conv4_neg *= torch.clip(torch.normal(1, sigma, size=Conv4_neg.size()).to(device), 0, 100)
            Conv5_pos *= torch.clip(torch.normal(1, sigma, size=Conv5_pos.size()).to(device), 0, 100)
            Conv5_neg *= torch.clip(torch.normal(1, sigma, size=Conv5_neg.size()).to(device), 0, 100)
            Conv6_pos *= torch.clip(torch.normal(1, sigma, size=Conv6_pos.size()).to(device), 0, 100)
            Conv6_neg *= torch.clip(torch.normal(1, sigma, size=Conv6_neg.size()).to(device), 0, 100)
            Conv7_pos *= torch.clip(torch.normal(1, sigma, size=Conv7_pos.size()).to(device), 0, 100)
            Conv7_neg *= torch.clip(torch.normal(1, sigma, size=Conv7_neg.size()).to(device), 0, 100)
            FC1_pos *= torch.clip(torch.normal(1, sigma, size=FC1_pos.size()).to(device), 0, 100)
            FC1_neg *= torch.clip(torch.normal(1, sigma, size=FC1_neg.size()).to(device), 0, 100)
            FC2_pos *= torch.clip(torch.normal(1, sigma, size=FC2_pos.size()).to(device), 0, 100)
            FC2_neg *= torch.clip(torch.normal(1, sigma, size=FC2_neg.size()).to(device), 0, 100)

            # weight is calculated through (G+ - G-)
            model_test.Conv1.weight = nn.Parameter(Conv1_pos - Conv1_neg)
            model_test.Conv2.weight = nn.Parameter(Conv2_pos - Conv2_neg)
            model_test.Conv3.weight = nn.Parameter(Conv3_pos - Conv3_neg)
            model_test.Conv4.weight = nn.Parameter(Conv4_pos - Conv4_neg)
            model_test.Conv5.weight = nn.Parameter(Conv5_pos - Conv5_neg)
            model_test.Conv6.weight = nn.Parameter(Conv6_pos - Conv6_neg)
            model_test.Conv7.weight = nn.Parameter(Conv7_pos - Conv7_neg)
            model_test.FC1.weight = nn.Parameter(FC1_pos - FC1_neg)
            model_test.FC2.weight = nn.Parameter(FC2_pos - FC2_neg)

        correct = 0
        total_cnt = 0
        valid_loss = 0

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
            print(f"\nValid Acc After OffChip Transfer : {valid_acc}")
            print(f"Valid Loss After OffChip Transfer : {valid_loss / total_cnt}")

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
                batch[0], batch[1] = batch[0].to(device), batch[1].to(device)

                logits = model_test(batch[0])
                loss = loss_fn(logits, batch[1])
                loss.backward()

                optimizer.step()

                #   LTPLTD Reflection
                with torch.no_grad():
                    if Parameter.Onchip_ParMode == 'LTPLTD_both':
                        Time = (Parameter.FullTimeLTP + Parameter.FUllTimeLTD) / 2
                    if Parameter.Onchip_ParMode == 'LTP_only':
                        Time = Parameter.FullTimeLTP
                    if Parameter.Onchip_ParMode == 'LTD_only':
                        Time = Parameter.FullTimeLTP + Parameter.FUllTimeLTD



                    dwf = model_test.Conv1.weight - (Conv1_pos - Conv1_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(Conv1_pos, Conv1_neg, dwf, Parameter.LTP, Parameter.LTD)
                    Conv1_pos += dwfpos
                    Conv1_neg += dwfneg
                    Conv1_pos.clamp_(Min1, 1)
                    Conv1_neg.clamp_(Min1, 1)
                    model_test.Conv1.weight = nn.Parameter((Conv1_pos - Conv1_neg))

                    dwf = model_test.Conv2.weight - (Conv2_pos - Conv2_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(Conv2_pos, Conv2_neg, dwf, Parameter.LTP, Parameter.LTD)
                    Conv2_pos += dwfpos
                    Conv2_neg += dwfneg
                    Conv2_pos.clamp_(Min1, 1)
                    Conv2_neg.clamp_(Min1, 1)
                    model_test.Conv2.weight = nn.Parameter((Conv2_pos - Conv2_neg))

                    dwf = model_test.Conv3.weight - (Conv3_pos - Conv3_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(Conv3_pos, Conv3_neg, dwf, Parameter.LTP, Parameter.LTD)
                    Conv3_pos += dwfpos
                    Conv3_neg += dwfneg
                    Conv3_pos.clamp_(Min1, 1)
                    Conv3_neg.clamp_(Min1, 1)
                    model_test.Conv3.weight = nn.Parameter((Conv3_pos - Conv3_neg))

                    dwf = model_test.Conv4.weight - (Conv4_pos - Conv4_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(Conv4_pos, Conv4_neg, dwf, Parameter.LTP, Parameter.LTD)
                    Conv4_pos += dwfpos
                    Conv4_neg += dwfneg
                    Conv4_pos.clamp_(Min1, 1)
                    Conv4_neg.clamp_(Min1, 1)
                    model_test.Conv4.weight = nn.Parameter((Conv4_pos - Conv4_neg))

                    dwf = model_test.Conv5.weight - (Conv5_pos - Conv5_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(Conv5_pos, Conv5_neg, dwf, Parameter.LTP, Parameter.LTD)
                    Conv5_pos += dwfpos
                    Conv5_neg += dwfneg
                    Conv5_pos.clamp_(Min1, 1)
                    Conv5_neg.clamp_(Min1, 1)
                    model_test.Conv5.weight = nn.Parameter((Conv5_pos - Conv5_neg))

                    dwf = model_test.Conv6.weight - (Conv6_pos - Conv6_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(Conv6_pos, Conv6_neg, dwf, Parameter.LTP, Parameter.LTD)
                    Conv6_pos += dwfpos
                    Conv6_neg += dwfneg
                    Conv6_pos.clamp_(Min1, 1)
                    Conv6_neg.clamp_(Min1, 1)
                    model_test.Conv6.weight = nn.Parameter((Conv6_pos - Conv6_neg))

                    dwf = model_test.Conv7.weight - (Conv7_pos - Conv7_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(Conv7_pos, Conv7_neg, dwf, Parameter.LTP, Parameter.LTD)
                    Conv7_pos += dwfpos
                    Conv7_neg += dwfneg
                    Conv7_pos.clamp_(Min1, 1)
                    Conv7_neg.clamp_(Min1, 1)
                    model_test.Conv7.weight = nn.Parameter((Conv7_pos - Conv7_neg))

                    dwf = model_test.FC1.weight - (FC1_pos - FC1_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(FC1_pos, FC1_neg, dwf, Parameter.LTP, Parameter.LTD)
                    FC1_pos += dwfpos
                    FC1_neg += dwfneg
                    FC1_pos.clamp_(Min2, 1)
                    FC1_neg.clamp_(Min2, 1)
                    model_test.FC1.weight = nn.Parameter((FC1_pos - FC1_neg))

                    dwf = model_test.FC2.weight - (FC2_pos - FC2_neg)
                    dwf = quant_dev_update(dwf)
                    UpdateCurrEnergy += torch.abs(dwf).sum() * Parameter.HighVoltage * Time * Parameter.WriteCurr
                    dwfpos, dwfneg = ltpltd(FC2_pos, FC2_neg, dwf, Parameter.LTP, Parameter.LTD)
                    FC2_pos += dwfpos
                    FC2_neg += dwfneg
                    FC2_pos.clamp_(Min3, 1)
                    FC2_neg.clamp_(Min3, 1)
                    model_test.FC2.weight = nn.Parameter((FC2_pos - FC2_neg))

                optimizer.zero_grad()

                class_init = 0
                train_loss += loss.item()
                _, predict = logits.max(1)

                total_cnt += batch[1].size(0)
                correct += predict.eq(batch[1]).sum().item()



                if step % 200 == 0:
                    print(f"\n====== {step} Step of {len(train_loader)} ======")
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
                        print(f"\nValid Acc during Onchip_Parallel: {valid_acc}")
                        print(f"Valid Loss during Onchip_Parallel : {valid_loss / total_cnt}")

                if step >= Parameter.DATANUM:
                    break


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
        self.DROP1 = nn.Dropout2d(p = 0)
        self.DROP2 = nn.Dropout(p = 0)


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
        out = self.FC2(out) * 2

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
        global ReadSynEnergy
        global UpdateCapEnergy
        pweight = torch.where(self.weight >=0, self.weight, self.weight * 0)
        nweight = torch.where(self.weight < 0, self.weight * -1, self.weight * 0)
        if input.min() >= 0:
            input = quant_dev.apply(input, (2**Parameter.InputPrecesion), Parameter.ADCMax)

            Level = (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.DelPrecision/Parameter.DelEn)))
            Outpos = F.conv2d(quant_dev_back.apply(input, Level), pweight, self.bias, self.stride, self.padding)
            Outneg = F.conv2d(quant_dev_back.apply(input, Level), nweight, self.bias, self.stride, self.padding)
            with torch.no_grad():
                if input.size(dim=0) == 1:
                    ReadSynEnergy += Outpos.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
                    ReadSynEnergy += Outneg.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
                if input.size(dim = 0) == 1:
                    HighVVI = (Parameter.HighVoltage ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                    # There are G+, G- and activity factor is included
                    Act = 0.1 * 2
                    if Parameter.Onchip_ParMode == 'LTPLTD_both':
                        Act *= 0.5      #Energy normalized for discharge and charge process
                        Act *= 2        # There is LTP LTD process for both G+ and G-
                        UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act
                    if Parameter.Onchip_ParMode == 'LTP_only':
                        Act *= 0.75  # Energy normalized for discharge and charge process
                        UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act
                    if Parameter.Onchip_ParMode == 'LTD_only':
                        Act *= 0.25  # Energy normalized for discharge and charge process
                        UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act


            Outpos = quant_dev.apply(Outpos, (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)
            Outneg = quant_dev.apply(Outneg, (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)

            Out = (Outpos - Outneg)
            Out = quant_dev_back_input.apply(Out)
            return Out

        if input.min() < 0:
            inpos = torch.where(input >= 0, input, input *0)
            inneg = torch.where(input < 0, input * -1, input * 0)
            inpos = quant_dev.apply(inpos, (2**Parameter.InputPrecesion), 2.7537)
            inneg = quant_dev.apply(inneg, (2**Parameter.InputPrecesion), 2.7537)
            Level = (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.DelPrecision/Parameter.DelEn)))
            Outp1 = F.conv2d(quant_dev_back.apply(inpos, Level), pweight, self.bias, self.stride, self.padding)
            Outp2 = F.conv2d(quant_dev_back.apply(inneg, Level), nweight, self.bias, self.stride, self.padding)
            Outn1 = F.conv2d(quant_dev_back.apply(inneg, Level), pweight, self.bias, self.stride, self.padding)
            Outn2 = F.conv2d(quant_dev_back.apply(inpos, Level), nweight, self.bias, self.stride, self.padding)
            with torch.no_grad():
                if input.size(dim=0) == 1:
                    ReadSynEnergy += Outp1.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
                    ReadSynEnergy += Outn1.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
                    ReadSynEnergy += Outp2.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
                    ReadSynEnergy += Outn2.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
                if input.size(dim = 0) == 1:
                    HighVVI = (Parameter.HighVoltage ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                    # There are G+, G- and activity factor is included, Input is both positive and negative
                    Act = 0.1 * 2 * 2
                    if Parameter.Onchip_ParMode == 'LTPLTD_both':
                        Act *= 0.5      #Energy normalized for discharge and charge process
                        Act *= 2        # There are both LTP LTD process
                        UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act
                    if Parameter.Onchip_ParMode == 'LTP_only':
                        Act *= 0.75  # Energy normalized for discharge and charge process
                        UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act
                    if Parameter.Onchip_ParMode == 'LTD_only':
                        Act *= 0.25  # Energy normalized for discharge and charge process
                        UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act

            Outp1 = quant_dev.apply(Outp1, (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)
            Outp2 = quant_dev.apply(Outp2, (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)
            Outn1 = quant_dev.apply(Outn1, (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)
            Outn2 = quant_dev.apply(Outn2, (self.kernel_size[0] ** 2) * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)

            Out = Outp1 + Outp2 - Outn1 - Outn2
            Out = quant_dev_back_input.apply(Out)
            return Out


class HardwareLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(HardwareLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        global ReadSynEnergy
        global UpdateCapEnergy
        input = quant_dev.apply(input, (2**Parameter.InputPrecesion), Parameter.ADCMax)
        pweight = torch.where(self.weight >=0, self.weight, self.weight * 0)
        nweight = torch.where(self.weight < 0, self.weight * -1, self.weight * 0)
        Added = math.ceil(self.in_features / Parameter.SubRow)
        Level = Added * (2 ** (Parameter.ADCPrecision + math.ceil(Parameter.DelPrecision / Parameter.DelEn)))
        Outpos = F.linear(quant_dev_back.apply(input, Level), pweight)
        Outneg = F.linear(quant_dev_back.apply(input, Level), nweight)
        with torch.no_grad():
            SIZECOL = Outpos.size(dim=1)
            if input.size(dim=0) == 1:
                ReadSynEnergy += Outpos.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
                ReadSynEnergy += Outneg.sum() * Parameter.ONCurrent * Parameter.ReadTimePerCycle * Parameter.VRead
            if input.size(dim=0) == 1:
                HighVVI = (Parameter.HighVoltage ** 2)  * Parameter.InputPrecesion * math.ceil(SIZECOL/Parameter.SubCol)
                # Activity Factor and G+ G-
                Act = 0.1 * 2       # Activity Factor included
                if Parameter.Onchip_ParMode == 'LTPLTD_both':
                    Act *= 0.5
                    Act *= 2
                    UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act
                if Parameter.Onchip_ParMode == 'LTP_only':
                    Act *= 0.75
                    UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act
                if Parameter.Onchip_ParMode == 'LTD_only':
                    Act *= 0.25
                    UpdateCapEnergy += torch.count_nonzero(input) * Parameter.SubCol * 0.06e-15 * HighVVI * Act

        Outpos = quant_dev.apply(Outpos, Added * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)
        Outneg = quant_dev.apply(Outneg, Added * (2**(Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding))), Parameter.ADCMax)

        Out = Outpos - Outneg
        if Out.size()[1] != Parameter.NUMCLASS:
            Out = quant_dev_back_input.apply(Out)
        if Out.size()[1] == Parameter.NUMCLASS:
            Out = quant_dev_back_input_last.apply(Out)

        return Out

def quant_dev_weight(x, level):
    x_ = x.clone()
    # index_separator = torch.linspace(0, torch.max(x_).item(), level).to(device)
    index_separator = torch.linspace(0, 1, level).to(device)
    # n = torch.normal(0, 0.3, size=x_.shape).to(device) * index_separator[1]
    x_ = index_separator[torch.searchsorted(index_separator, x_ + index_separator[1] / 2) - 1]
    # x_ = x_.clip(min=0, max=7)
    return x_

def quant_dev_update(x):
    x_ = x.clone()
    #        index_separator = torch.linspace(0, torch.max(x_).item(), level).to(device)
    x_ = x_.clip(min=-Parameter.UpdateMax, max=Parameter.UpdateMax)
    index_separator = torch.linspace(-Parameter.UpdateMax, Parameter.UpdateMax, Parameter.UpdateLEVELs + 1).to(device)
    # n = torch.normal(0, 0.3, size=x_.shape).to(device) * index_separator[1]
    x_ = index_separator[torch.searchsorted(index_separator, x_)]
    #        x_ = x_.clip(min=0, max=7)
    return x_

class quant_dev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, level, Max):
        ctx.set_materialize_grads(False)
        x_ = x.clone()
        #index_separator = torch.linspace(0, torch.max(x_).item(), level).to(device)
        index_separator = torch.linspace(0, Max, level).to(device)
        # n = torch.normal(0, 0.3, size=x_.shape).to(device) * index_separator[1]
        x_ = index_separator[torch.searchsorted(index_separator, x_ + index_separator[1] / 2) - 1]
        #x_ = x_.clip(min=0, max=7)
        return x_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class quant_dev_back(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Level):
        ctx.set_materialize_grads(False)
        ctx.Level = Level
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x_ = grad_output.clone()
        Level = ctx.Level
        x_ = x_.clip(min=-Parameter.DelADCMax, max=Parameter.DelADCMax)
        index_separator = torch.linspace(-Parameter.DelADCMax, Parameter.DelADCMax, Level).to(device)
        x_ = index_separator[torch.searchsorted(index_separator, x_)]
        return x_, None, None

class quant_dev_back_input(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        global UpdateCapEnergy
        x_ = grad_output.clone()
        x_ = x_.clip(min=-Parameter.DelADCMax, max=Parameter.DelADCMax)
        index_separator = torch.linspace(-Parameter.DelADCMax, Parameter.DelADCMax,
                                         2 ** (Parameter.DelPrecision+1) +1 ).to(device)
        x_ = index_separator[torch.searchsorted(index_separator, x_)]
        if x_.size(dim=0) == 1:
            HighVVIinh = (Parameter.HighVoltage ** 2) * Parameter.UPDEL
            # Activity factor, G+ G-
            Act = 0.1 * 2
            if x_.dim() >2:
                Act *= Parameter.Kernel ** 2
            if x_.dim() <= 2:
                if Parameter.LayerType[Parameter.LayerInChan.index(x_.size(dim = 1))-1] == 'FC':
                    Act *= Parameter.LayerInChan[Parameter.LayerInChan.index(x_.size(dim = 1))-1] / Parameter.SubRow

            if Parameter.Onchip_ParMode == 'LTPLTD_both':
                Act *= 0.5
                Act *= 2
                UpdateCapEnergy += torch.count_nonzero(x_) * Parameter.SubRow * 0.08e-15 * HighVVIinh * Act
            if Parameter.Onchip_ParMode == 'LTP_only':
                Act *= 0.25
                UpdateCapEnergy += torch.count_nonzero(x_) * Parameter.SubRow * 0.08e-15 * HighVVIinh * Act
            if Parameter.Onchip_ParMode == 'LTD_only':
                Act *= 0.75
                UpdateCapEnergy += torch.count_nonzero(x_) * Parameter.SubRow * 0.08e-15 * HighVVIinh * Act

        return x_, None, None

class quant_dev_back_input_last(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        global UpdateCapEnergy
        x_ = grad_output.clone()
        x_ = x_.clip(min=-15 * Parameter.DelADCMax, max=15 * Parameter.DelADCMax)
        index_separator = torch.linspace(-15 * Parameter.DelADCMax, 15 * Parameter.DelADCMax,
                                         2 ** (Parameter.DelPrecision+1) +1 ).to(device)
        x_ = index_separator[torch.searchsorted(index_separator, x_)]
        if x_.size(dim=0) == 1:
            HighVVIinh = (Parameter.HighVoltage ** 2) * Parameter.UPDEL
            # Activity factor, G+ G-
            Act = 0.1 * 2
            Act *= Parameter.LayerInChan[Parameter.LayerInChan.index(x_.size(dim = 1))-1] / Parameter.SubRow

            if Parameter.Onchip_ParMode == 'LTPLTD_both':
                Act *= 0.5
                Act *= 2
                UpdateCapEnergy += torch.count_nonzero(x_) * Parameter.SubRow * 0.08e-15 * HighVVIinh * Act
            if Parameter.Onchip_ParMode == 'LTP_only':
                Act *= 0.25
                UpdateCapEnergy += torch.count_nonzero(x_) * Parameter.SubRow * 0.08e-15 * HighVVIinh * Act
            if Parameter.Onchip_ParMode == 'LTD_only':
                Act *= 0.75
                UpdateCapEnergy += torch.count_nonzero(x_) * Parameter.SubRow * 0.08e-15 * HighVVIinh * Act

        return x_, None, None

def step_decay(optimizer, epoch, lr_init, decay_factor, step_size):
    lr = lr_init * math.pow(decay_factor, math.floor((1 + epoch) / step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def ltpltd (gp0, gn0, delta, ltpNL, ltdNL) :
    delta_positive = delta.clamp(min=0)
    delta_negative = -delta.clamp(max=0)

    if Parameter.Onchip_ParMode == 'LTPLTD_both':
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

    if Parameter.Onchip_ParMode == 'LTD_only':

        if ltpNL == 0:
            delta_positive_LTPupdated = 0
            delta_negative_LTPupdated = 0

        if ltpNL != 0:
            LTP_alpha = 1 / (1 - math.exp(-ltpNL))
            # delta_positive_LTPupdated : G+를 LTP 해줌
            delta_positive_LTPupdated = 0
            # delta_negative_LTPupdated : G-를 LTP 해줌 (이 결과는 양수임)
            delta_negative_LTPupdated = 0

        if ltdNL == 0:
            delta_positive_LTDupdated = -1 * delta_positive
            delta_negative_LTDupdated = -1 * delta_negative

        if ltdNL != 0:
            LTD_alpha = 1 / (1 - math.exp(-ltdNL))
            # delta_positive_LTPupdated : G-를 LTD 해줌 (이 결과는 음수임)
            delta_positive_LTDupdated = 1 * delta_positive * (-ltdNL * (LTD_alpha - 1 + gn0))
            # delta_negative_LTDupdated : G+를 LTD 해줌 (이 결과는 음수임, delta_negative가 양수라서)
            delta_negative_LTDupdated = 1 * delta_negative * (-ltdNL * (LTD_alpha - 1 + gp0))

    if Parameter.Onchip_ParMode == 'LTP_only':
        if ltpNL == 0:
            delta_positive_LTPupdated = 1 * delta_positive
            delta_negative_LTPupdated = 1 * delta_negative

        if ltpNL != 0:
            LTP_alpha = 1 / (1 - math.exp(-ltpNL))
            # delta_positive_LTPupdated : G+를 LTP 해줌
            delta_positive_LTPupdated = 1 * delta_positive * (ltpNL * (LTP_alpha - gp0))
            # delta_negative_LTPupdated : G-를 LTP 해줌 (이 결과는 양수임)
            delta_negative_LTPupdated = 1 * delta_negative * (ltpNL * (LTP_alpha - gn0))

        if ltdNL == 0:
            delta_positive_LTDupdated = 0
            delta_negative_LTDupdated = 0

        if ltdNL != 0:
            LTD_alpha = 1 / (1 - math.exp(-ltdNL))
            # delta_positive_LTPupdated : G-를 LTD 해줌 (이 결과는 음수임)
            delta_positive_LTDupdated = 0
            # delta_negative_LTDupdated : G+를 LTD 해줌 (이 결과는 음수임, delta_negative가 양수라서)
            delta_negative_LTDupdated = 0

    return (delta_positive_LTPupdated-delta_positive_LTDupdated), (delta_negative_LTPupdated - delta_negative_LTDupdated)


# Xavier
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)

# sys.stdout.close()