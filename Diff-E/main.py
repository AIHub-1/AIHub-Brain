from models import *
from utils import *

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

# Evaluate function
def evaluate(encoder, fc, generator, device):
    labels = np.arange(0, 13)
    Y = []
    Y_hat = []
    for x, y in generator:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        encoder_out = encoder(x)
        y_hat = fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)

        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

    # List of tensors to tensor to numpy
    Y = torch.cat(Y, dim=0).numpy()  # (N, )
    Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

    # Accuracy and Confusion Matrix
    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
    f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "auc": auc,
    }
    # df_cm = pd.DataFrame(confusion_matrix(Y, Y_hat.argmax(axis=1)))
    return metrics


def train(args):
    subject = args.subject
    device = args.device
    device = torch.device(device)
    batch_size = 32
    batch_size2 = 260
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # EEG data path
    root_dir = "Path-to-the-data"
    # Write performance metrics to file
    # output_dir = "performance-metric-path"
    # output_file = f"{output_dir}/{subject}.txt"

    # Load data
    X, Y = load_data(root_dir=root_dir, subject=subject, session=1)
    # Dataloader
    train_loader, test_loader = get_dataloader(
        X, Y, batch_size, batch_size2, seed, shuffle=True
    )

    # Define model
    num_classes = 13
    channels = X.shape[1]

    n_T = 1000
    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512

    ddpm_model = ConditionalUNet(in_channels=channels, n_feat=ddpm_dim).to(device)
    ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(
        device
    )
    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    decoder = Decoder(
        in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
    ).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    print("ddpm size: ", sum(p.numel() for p in ddpm.parameters()))
    print("encoder size: ", sum(p.numel() for p in encoder.parameters()))
    print("decoder size: ", sum(p.numel() for p in decoder.parameters()))
    print("fc size: ", sum(p.numel() for p in fc.parameters()))

    # Criterion
    criterion = nn.L1Loss()
    criterion_class = nn.MSELoss()

    # Define optimizer
    base_lr, lr = 9e-5, 1.5e-3
    optim1 = optim.RMSprop(ddpm.parameters(), lr=base_lr)
    optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)

    # EMAs
    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10,)

    step_size = 150
    scheduler1 = optim.lr_scheduler.CyclicLR(
        optimizer=optim1,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    scheduler2 = optim.lr_scheduler.CyclicLR(
        optimizer=optim2,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    # Train & Evaluate
    num_epochs = 500
    test_period = 1
    start_test = test_period
    alpha = 0.1

    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_auc = 0

    with tqdm(
        total=num_epochs, desc=f"Method ALL - Processing subject {subject}"
    ) as pbar:
        for epoch in range(num_epochs):
            ddpm.train()
            diffe.train()

            ############################## Train ###########################################
            for x, y in train_loader:
                x, y = x.to(device), y.type(torch.LongTensor).to(device)
                y_cat = F.one_hot(y, num_classes=13).type(torch.FloatTensor).to(device)
                # Train DDPM
                optim1.zero_grad()
                x_hat, down, up, noise, t = ddpm(x)

                loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                loss_ddpm.mean().backward()
                optim1.step()
                ddpm_out = x_hat, down, up, t

                # Train Diff-E
                optim2.zero_grad()
                decoder_out, fc_out = diffe(x, ddpm_out)

                loss_gap = criterion(decoder_out, loss_ddpm.detach())
                loss_c = criterion_class(fc_out, y_cat)
                loss = loss_gap + alpha * loss_c
                loss.backward()
                optim2.step()

                # Optimizer scheduler step
                scheduler1.step()
                scheduler2.step()

                # EMA update
                fc_ema.update()

            ############################## Test ###########################################
            with torch.no_grad():
                if epoch > start_test:
                    test_period = 1
                if epoch % test_period == 0:
                    ddpm.eval()
                    diffe.eval()

                    metrics_test = evaluate(diffe.encoder, fc_ema, test_loader, device)

                    acc = metrics_test["accuracy"]
                    f1 = metrics_test["f1"]
                    recall = metrics_test["recall"]
                    precision = metrics_test["precision"]
                    auc = metrics_test["auc"]

                    best_acc_bool = acc > best_acc
                    best_f1_bool = f1 > best_f1
                    best_recall_bool = recall > best_recall
                    best_precision_bool = precision > best_precision
                    best_auc_bool = auc > best_auc

                    if best_acc_bool:
                        best_acc = acc
                        # torch.save(diffe.state_dict(), f'./models/diffe_{subject}.pt')
                    if best_f1_bool:
                        best_f1 = f1
                    if best_recall_bool:
                        best_recall = recall
                    if best_precision_bool:
                        best_precision = precision
                    if best_auc_bool:
                        best_auc = auc

                    # print("Subject: {0}".format(subject))
                    # # print("ddpm test loss: {0:.4f}".format(t_test_loss_ddpm/len(test_generator)))
                    # # print("encoder test loss: {0:.4f}".format(t_test_loss_ed/len(test_generator)))
                    # print("accuracy:  {0:.2f}%".format(acc*100), "best: {0:.2f}%".format(best_acc*100))
                    # print("f1-score:  {0:.2f}%".format(f1*100), "best: {0:.2f}%".format(best_f1*100))
                    # print("recall:    {0:.2f}%".format(recall*100), "best: {0:.2f}%".format(best_recall*100))
                    # print("precision: {0:.2f}%".format(precision*100), "best: {0:.2f}%".format(best_precision*100))
                    # print("auc:       {0:.2f}%".format(auc*100), "best: {0:.2f}%".format(best_auc*100))
                    # writer.add_scalar(f"EEGNet/Accuracy/subject_{subject}", acc*100, epoch)
                    # writer.add_scalar(f"EEGNet/F1-score/subject_{subject}", f1*100, epoch)
                    # writer.add_scalar(f"EEGNet/Recall/subject_{subject}", recall*100, epoch)
                    # writer.add_scalar(f"EEGNet/Precision/subject_{subject}", precision*100, epoch)
                    # writer.add_scalar(f"EEGNet/AUC/subject_{subject}", auc*100, epoch)

                    # if best_acc_bool or best_f1_bool or best_recall_bool or best_precision_bool or best_auc_bool:
                    #     performance = {'subject': subject,
                    #                 'epoch': epoch,
                    #                 'accuracy': best_acc*100,
                    #                 'f1_score': best_f1*100,
                    #                 'recall': best_recall*100,
                    #                 'precision': best_precision*100,
                    #                 'auc': best_auc*100
                    #                 }
                    #     with open(output_file, 'a') as f:
                    #         f.write(f"{performance['subject']}, {performance['epoch']}, {performance['accuracy']}, {performance['f1_score']}, {performance['recall']}, {performance['precision']}, {performance['auc']}\n")
                    description = f"Best accuracy: {best_acc*100:.2f}%"
                    pbar.set_description(
                        f"Method ALL - Processing subject {subject} - {description}"
                    )
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model")
    # Define command-line arguments
    parser.add_argument(
        "--num_subjects", type=int, default=22, help="number of subjects to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)"
    )

    # Parse command-line arguments
    args = parser.parse_args()
    for i in range(2, args.num_subjects + 1):
        subject = i
        args.subject = subject
        train(args)
