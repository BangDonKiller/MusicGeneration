import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import MusicVAE
from generate_dataset import inputs, targets, test_samples, mappings_length


def reconstruction_loss(output, target):
    return F.cross_entropy(output, target)


def kl_divergence_loss(mu, sigma, free_bits=33.3):
    kl_loss = -0.5 * torch.sum(
        1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2), dim=1
    )
    kl_loss = torch.clamp(kl_loss - free_bits, min=0)
    return kl_loss.mean()


def beta_growth(epoch):
    start = 0.0
    end = 0.01
    k = 0.9
    x0 = 10
    return (end - start) / (1 + math.exp(-k * (epoch - x0))) + start


def calculate_accuracy(output, target):
    output = torch.argmax(output, dim=2)
    target = torch.argmax(target, dim=2)

    # print("output: ", output)
    # print("target: ", target)
    accuracy = torch.mean((output == target).float())
    return accuracy.item()


# 訓練函數
def train_vae(
    model,
    dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    teacher_forcing=True,
    train=True,
    autoencoder=True,
):
    model.to(device)
    model.train()

    global_step = 0  # 用於 Scheduled Sampling 計算
    # max_beta = 0.2  # 最大β值
    beta = 0.0001  # 初始β值
    beta_growth_rate = 1.099999  # β增長的指數率

    recons_loss = []
    KLDIV_loss = []
    tot_loss = []
    total_accuracy = []
    tot_accuracy = []

    for epoch in range(num_epochs):
        total_loss = 0
        tot_recons_loss = 0
        tot_KLDIV_loss = 0
        total_accuracy = 0

        # if epoch >= 10:
        #     teacher_forcing = False

        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if autoencoder:
                output = model(
                    x, y, teacher_forcing=teacher_forcing, train=train, epoch=epoch
                )
            else:
                output, mu, sigma = model(
                    x, y, teacher_forcing=teacher_forcing, train=train
                )

            # 計算重構損失和KL散度損失
            recon_loss = reconstruction_loss(output, y)

            if autoencoder:
                kl_loss = torch.zeros(1).cuda()
            else:
                kl_loss = kl_divergence_loss(mu, sigma)
            loss = recon_loss + beta * kl_loss

            # 計算準確度
            accuracy = calculate_accuracy(output, y)

            # 反向傳播
            loss.backward()
            optimizer.step()

            tot_KLDIV_loss += kl_loss.item()
            tot_recons_loss += recon_loss.item()
            total_loss += loss.item()
            total_accuracy += accuracy

            # 更新學習率
            scheduler.step()

            # 更新 β
            # beta = min(max_beta, beta * beta_growth_rate)  # β的指數增長
            beta = beta_growth(epoch)
            global_step += 1

        tot_loss.append(total_loss)
        recons_loss.append(tot_recons_loss)
        KLDIV_loss.append(tot_KLDIV_loss)
        tot_accuracy.append(total_accuracy / len(dataloader))

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}, Accuracy: {total_accuracy / len(dataloader)}"
        )

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./model weight/epoch{epoch}.pth")

    return recons_loss, KLDIV_loss, tot_loss, total_accuracy


# 準備數據
def prepare_data(train_data_tensor, target_data_tensor, test_data_tensor, batch_size):
    train_dataset = TensorDataset(train_data_tensor, target_data_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_data_tensor)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return dataloader, eval_loader


if __name__ == "__main__":
    # 設定參數
    input_dim = mappings_length
    lstm_hidden_dim = 512
    latent_dim = 32
    conductor_hidden_dim = 512
    conductor_output_dim = 256
    decoder_hidden_dim = 1024
    output_dim = input_dim
    batch_size = 32
    num_epochs = 50
    initial_learning_rate = 1e-3  # 初始學習率
    final_learning_rate = 1e-5  # 最終學習率

    autoencoder = True
    teacher_forcing = False
    TRAIN = False
    # 確定使用的設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型與優化器
    model = MusicVAE(
        input_dim,
        lstm_hidden_dim,
        latent_dim,
        conductor_hidden_dim,
        conductor_output_dim,
        decoder_hidden_dim,
        output_dim,
        batch_size,
        mappings_length,
        autoencoder,
    )
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # 定義學習率衰減調度器
    scheduler = ExponentialLR(optimizer, gamma=0.9999)  # 學習率以0.9999的速率指數衰減

    data_tensor = inputs  # 這裡的inputs_tensor作為自動重建的輸入和目標
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

    target_tensor = targets
    target_tensor = torch.tensor(target_tensor, dtype=torch.float32)

    test_tensor = test_samples[:256]
    test_tensor = torch.tensor(test_tensor, dtype=torch.float32)

    # 準備數據集和DataLoader
    dataloader, eval_loader = prepare_data(
        data_tensor, target_tensor, test_tensor, batch_size
    )

    # 訓練模型
    if TRAIN:
        recon_loss, KL_loss, total_loss, total_accuracy = train_vae(
            model,
            dataloader,
            optimizer,
            scheduler,
            num_epochs,
            device,
            teacher_forcing,
            train=TRAIN,
            autoencoder=autoencoder,
        )

        # 儲存模型
        torch.save(model.state_dict(), f"./model weight/epoch{num_epochs}.pth")

    else:
        model.load_state_dict(torch.load(f"./model weight/epoch{num_epochs}.pth"))

        # 生成樂曲
        model.eval()
        model.to(device)

        total_accuracy = 0
        total_loss = 0
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            if autoencoder:
                output = model(
                    x, y, teacher_forcing=teacher_forcing, train=TRAIN, epoch=None
                )
            else:
                output, mu, sigma = model(
                    x, y, teacher_forcing=teacher_forcing, train=TRAIN, epoch=None
                )

            # 計算重構損失和KL散度損失
            recon_loss = reconstruction_loss(output, y)

            if autoencoder:
                kl_loss = torch.zeros(1).cuda()
            else:
                kl_loss = kl_divergence_loss(mu, sigma)
            loss = recon_loss + kl_loss

            total_loss += loss.item()

            accuracy = calculate_accuracy(output, y)
            total_accuracy += accuracy

        print(f"Accuracy: {total_accuracy / len(eval_loader)}")
        print(f"Loss: {total_loss / len(eval_loader)}")
