import torchvision
import torchvision.transforms as tt
import torch
import pickle
import json
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from torch_topological.nn import VietorisRipsComplex
from copy import deepcopy
from datetime import datetime

model_training_dir = "experiment_results/models"
chart_output_dir = "experiment_results/charts_comparison"

if not os.path.exists(chart_output_dir):
    os.makedirs(chart_output_dir)

topology_stopped_dir = os.path.join(model_training_dir, "topology_stopped")
val_stopped_dir = os.path.join(model_training_dir, "val_stopped")


normalize_transform = tt.Compose([tt.ToTensor(), tt.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# Merge the train and testing sets to allow for sampling
cifar100_train  = torchvision.datasets.CIFAR100("../cifar100_data", train=True, download=True, transform=normalize_transform)
cifar100_test   = torchvision.datasets.CIFAR100("../cifar100_data", train=False, download=True, transform=normalize_transform)

cifar100 = torch.utils.data.ConcatDataset([cifar100_train, cifar100_test])
n_cifar = len(cifar100)


val_stopped = set(os.listdir(val_stopped_dir))
top_stopped = set(os.listdir(topology_stopped_dir))

models = val_stopped & top_stopped

def evaluate_model(model, dataset_loader):
    running_vloss = 0.0
    vacc_count = 0
    vacc_total = 0
    with torch.no_grad():
        for i, vdata in enumerate(dataset_loader):
            print(f"Calculating evaluation set {i}", end='\r')
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to('cuda'), vlabels.to('cuda')
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            vacc_count += count_correct_labels(voutputs, vlabels)
            vacc_total += len(vinputs)
            running_vloss += vloss
        print()
    return running_vloss/(i + 1), vacc_count/vacc_total

val_losses = []
max_losses = []
change_losses = []

val_accs = []
max_accs = []
change_accs = []

val_stop_epoch = []
max_stop_epoch = []
change_stop_epoch = []

for model_id in models:
    model_path_top = os.path.join(topology_stopped_dir, model_id)
    model_path_val = os.path.join(val_stopped_dir, model_id)

    samples_path = os.path.join(model_path_val, f"{model_id}_samples.json")

    with open(samples_path, 'r') as samples_f:
        samples = json.load(samples_f)    

    samples_test = samples['test']
    test_data = torch.utils.data.Subset(cifar100, samples_test)

    # load the best validation model
    best_val_path = os.path.join(model_path_val, f"{model_id}_best")
    best_val_model = torch.load(best_val_path)

    val_loss, val_acc = evaluate_model(best_val_model, test_data)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # load loss data
    loss_data_path = os.path.join(model_path_val, f"{model_id}_loss_data.json"):
    with open(loss_data_path, 'r') as loss_f:
        loss_data = json.load(loss_f)

    stop_epoch = list(loss_data.keys())[0]
    stop_loss = float("inf")
    for epoch, v in loss_data.item():
        if v[1] < stop_loss:
            stop_loss = v[1]
            stop_epoch = epoch
    val_stop_epoch.append(stop_epoch)

    # load topology data
    top_data_path = os.path.join(model_path_top, f"{model_id}_topology_data.json")
    with open(top_data_path, "r") as top_f:
        top_d = json.load(top_f)

    epochs = sorted([int(x) for x in top_d.keys()])
    top_vals = [top_d[str(v)] for v in epochs]

    # load the model with the highest homology number
    epoch_max_top = epochs[np.argmax(top_vals)]
    epoch_max_top_path = os.path.join(model_path_top, f"{model_id}_epoch_{epoch_max_top}")
    epoch_max_top_model = torch.load(epoch_max_top_path)

    max_top_loss, max_top_acc = evaluate_model(epoch_max_top_model, test_data)

    max_losses.apend(max_top_loss)
    max_accs.append(max_top_acc)
    max_stop_epoch.append(epoch_max_top)

    # laod the model with the highest topology change
    top_change = [abs(top_vals[i]-top_vals[i-1]) for i in epochs[1:]]
    epoch_max_change = epochs[np.argmax(top_vals)]
    epoch_max_change_path = os.path.join(model_path_top, f"{model_id}_epoch_{epoch_max_change}")
    epoch_max_change_model = torch.load(epoch_max_change_path)

    change_top_loss, change_top_acc = evaluate_model(epoch_max_change_model, test_data)

    change_losses.append(change_top_loss)
    change_accs.append(change_accs)
    change_stop_epoch.append(epoch_max_change)

df_loss = pd.DataFrame({
    'Validation': val_losses,
    'Maximum homology number': max_losses,
    'Maximum absolute change of homology number': change_losses
})

df_acc = pd.DataFrame({
    'Validation': val_accs,
    'Maximum homology number': max_accs,
    'Maximum absolute change of homology number': change_accs
})

fig, axs = plt.subplots(1, 2)

sns.boxplot(data=df_loss, ax=axs[0])
sns.boxplot(data=df_acc, ax=axs[1])

axs[0].set_xlabel("Early stopping method")
axs[0].set_ylabel("Test set loss")

axs[1].set_xlabel("Early stopping method")
axs[1].set_ylabel("Test set accuracy")

plt.suptitle("Comparison of early stopping methods - final model performance")

chart_output_path = os.path.join(chart_output_dir, "performance.png")
plt.savefig(chart_output_path)
plt.close()

df_time = pd.DataFrame({
    'Validation': val_stop_epoch,
    'Maximum homology number': max_stop_epoch,
    'Maximum absolute change of homology number': change_stop_epoch
})

sns.boxplot(data)
plt.xlabel("Early stopping method")
plt.ylabel("Epoch of early stopping occurance")
plt.suptitle("Comparison of early stopping methods - early stopping occurance")
chart_output_path = chart_output_path = os.path.join(chart_output_dir, "stop_time.png")
plt.savefig(chart_output_path)
plt.close()

fig, axs = plt.subplots(1, 2)
sns.regplot(x=val_stop_epoch, y=max_stop_epoch, ax=axs[0])
axs[0].set_xlabel("Validation method stop epoch")
axs[0].set_ylabel("Maximum homology number stop epoch")
axs[0].set_title("Correlation of early stopping epoch for different methods")

sns.regplot(x=val_stop_epoch, y=change_stop_epoch, ax=axs[1])
axs[1].set_xlabel("Validation method stop epoch")
axs[1].set_ylabel("Maximum homology change stop epoch")
axs[1].set_title("Correlation of early stopping epoch for different methods")
plt.suptitle("Comparison of early stopping methods - early stopping correlation time")
chart_output_path = chart_output_path = os.path.join(chart_output_dir, "correlation.png")
plt.savefig(chart_output_path)
plt.close()




