import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import math
import numpy as np
import os
from tqdm import tqdm
from model import MusicVAE
from loader import BatchGenerator

# from generate_dataset import inputs, targets, test_samples, mappings_length
from midi_util import read_midi_files


def beta_growth(epoch):
    start = 0.0
    end = 0.01
    k = 0.9
    x0 = 10
    return (end - start) / (1 + math.exp(-k * (epoch - x0))) + start


def calculate_accuracy(predict, target):
    return (predict == target).float().mean().item()


def create_log(check_path, model):
    count = 1

    check_path = path + str(count)
    while os.path.exists(check_path):
        count += 1
        check_path = path + str(count)

    os.mkdir(check_path)

    # model == True (AE) or model == False (VAE)
    if model:
        log_file = open(
            os.path.join(f"./model weight/log{count}/", "autoencoder.log"), "w+"
        )
    else:
        log_file = open(os.path.join(f"./model weight/log{count}/", "VAE.log"), "w+")
    print(
        "epoch,rec_loss,kld_loss,total_loss,teacher_forcing ratio,total_accuracy,val_loss,val_accuracy",
        file=log_file,
    )
    log_file.flush()

    return count, log_file


# 訓練函數
def train_vae(
    model,
    train_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    teacher_forcing=True,
    train=True,
    log_file=None,
    k=None,
):
    model.to(device)

    global_step = 0  # 用於 Scheduled Sampling 計算
    beta = 0.0001  # 初始β值

    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_rec = []
        total_kld = []
        total_acc = []

        for batch_idx, batch_song in tqdm(enumerate(train_loader())):
            batch_song = batch_song.to(device)
            batch_song = batch_song.permute(2, 0, 1)

            optimizer.zero_grad()
            output, song, mu, sigma = model(
                batch_song, truth=batch_song, teacher_forcing=teacher_forcing
            )

            recons_loss = model.reconstruction_loss(output, batch_song)
            KLDIV_loss = model.kl_divergence_loss(mu, sigma)
            if KLDIV_loss < 0:
                print("KLDIV_loss < 0")

            loss = recons_loss + beta * KLDIV_loss

            total_loss.append(loss.item())
            total_kld.append(KLDIV_loss.item())
            total_rec.append(recons_loss.item())
            total_acc.append(calculate_accuracy(song, batch_song))

            beta = beta_growth(batch_idx)

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

        print(
            "Epoch: {} | Reconstruction Loss: {:.4f} | KLD Loss: {:.4f} | Total Loss: {:.4f} | beta: {:.4f} | Accuracy: {:.4f}".format(
                epoch,
                np.mean(total_rec),
                np.mean(total_kld),
                np.mean(total_loss),
                beta,
                np.mean(total_acc),
            )
        )

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./model weight/epoch{epoch}.pth")

    torch.save(model.state_dict(), f"./model weight/epoch{epoch+1}.pth")

    return total_rec, total_kld, total_loss, total_acc


# 準備數據
def prepare_data(datass):
    train_loader = BatchGenerator(datass, batch_size=batch_size, shuffle=True)
    return train_loader


if __name__ == "__main__":
    # 設定參數
    input_dim = 512
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
    K = 25  # Scheduled Sampling 的 K 值

    teacher_forcing = True
    TRAIN = True

    # 確定使用的設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datass, targets, note_sets = read_midi_files(
        ["./dataset/mozart", "./dataset/Paganini"], None
    )
    note_size = (
        len(note_sets["timing"]),
        len(note_sets["duration"]),
        len(note_sets["pitch"]),
    )
    train_loader = prepare_data(datass)
    print("Data prepared.")

    # 初始化模型與優化器
    model = MusicVAE(
        input_dim,
        lstm_hidden_dim,
        latent_dim,
        conductor_hidden_dim,
        conductor_output_dim,
        decoder_hidden_dim,
        note_size,
    )
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # 定義學習率衰減調度器
    scheduler = ExponentialLR(optimizer, gamma=0.9999)  # 學習率以0.9999的速率指數衰減

    # 訓練模型
    if TRAIN:

        # path = "./model weight/log"
        # count, log_file = create_log(path, False)

        rec_loss, KL_loss, total_loss, total_acc = train_vae(
            model,
            train_loader,
            optimizer,
            scheduler,
            num_epochs,
            device,
            teacher_forcing,
            train=TRAIN,
            log_file=None,
            k=K,
        )

        print("Finished training.")

        # torch.save(
        #     model.state_dict(),
        #     f"./model weight/log{count}/VAE_epoch{num_epochs}.pth",
        # )
