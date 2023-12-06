import torchvision
import torchvision.transforms as tt
import torch
import pickle
import json
import os
import argparse
import numpy as np
import datetime
import torch_topological
from copy import deepcopy

def train_one_epoch(model, optimizer, epoch_index, tb_writer, training_loader, loss_fn):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        last_loss = running_loss / (i+1)
        print('  batch {} loss: {}'.format(i + 1, last_loss), end='\r')
    print("")
    return last_loss



# Merge the train and testing sets to allow for sampling
cifar100_train  = torchvision.datasets.CIFAR100("../cifar100_data", train=True, download=True)
cifar100_test   = torchvision.datasets.CIFAR100("../cifar100_data", train=False, download=True)

cifar100 = torch.utils.data.ConcatDataset([cifar100_train, cifar100_test])
n_cifar = len(cifar100)

N_MODELS=30
TRAIN_SIZE=38000
VAL_SIZE=12000
TEST_SIZE=10000
TRAIN_VAL_SIZE=TRAIN_SIZE+VAL_SIZE
BATCH_SIZE = 128

EPOCHS = 160
PATIENCE = 30

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--N_MODELS', default=1, type=int)
parser.add_argument('-b', '--BATCH_SIZE', default=128, type=int)
parser.add_argument('-e', '--EPOCHS', default=160, type=int)
parser.add_argument('-p', '--PATIENCE', default=300, type=int)
parser.add_argument('-s', '--SAVE_EVERY', default=1, type=int)

args = vars(parser.parse_args())

N_MODELS = args['N_MODELS']
BATCH_SIZE = args['BATCH_SIZE']

EPOCHS = args['EPOCHS']
PATIENCE = args['PATIENCE']
SAVE_EVERY = args['SAVE_EVERY']

model_base = torchvision.models.resnet50(num_classes=100, weights=None, zero_init_residual=True, norm_layer=None)
model_base.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model_base.maxpool = torch.nn.Identity() 

augment_transform = tt.Compose([
  tt.RandomCrop(size=32, padding=4),
  tt.RandomHorizontalFlip()
])

class AugmentedDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, transform):
    self.dataset=dataset
    self.transform=transform
  def __getitem__(self, idx):
    sample, target = self.dataset[idx]
    return self.transform(sample), target
  def __len__(self):
    return len(self.dataset)

normalize_transform = tt.Compose([tt.ToTensor(), tt.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

model_training_folder = "experiment_results/models"

def evaluate_h1_count(activations):
    rips_layer = torch_topological.nn.VietorisRipsComplex(dim=1, p=2)
    persistence_information = rips_layer(activations)
    dim_1 = persistence_information[1].diagram
    return len(dim_1)


def compute_mean_topology_count(activations, labels):
    sum_count = 0
    sum_classes = 0
    for cls in torch.unique(labels):
        activations_class = torch.select(activations, 0, labels==cls)
        count = evaluate_h1_count(activations_class)
        sum_count += count
        sum_classes += 1
    return sum_count/sum_classes


def compute_topology_count_for_model(model, train_loader):
    model = model.to('cuda')
    handle = model.layer4[-1].relu.register_forward_hook(activation_hook("stage_4", output=False, perform_max_pool=False))
    activation = {}
    def activation_hook(name, output=True, perform_max_pool=False):
        nonlocal activation
        if output:
            def hook(model, input, output):
                values = output.detach()
                if perform_max_pool:
                    # perform global average pooling
                    values = torch.mean(values, dim=(-2, -1), keepdim=False)
                activation[name] = values
        else:
            def hook(model, input, output):
                values = input[0].detach()
                if perform_max_pool:
                    # perform global average pooling
                    values = torch.mean(values, dim=(-2, -1), keepdim=False)
                activation[name] = values
        return hook
    
    activations = []
    labels = []

    for _, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        model(inputs)
        activations.append(activation["stage_4"].to('cpu'))
        labels.append(labels.to('cpu'))

    activations = torch.concatenate(activations, axis=0)
    labels = torch.concatenate(labels, axis=0)                        

    handle.remove()
    model = model.to('cpu')
    
    return compute_mean_topology_count(activations, labels)

def count_correct_labels(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.sum((classes == labels)).float()

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
    return running_vloss, vacc_count/vacc_total

if __name__ == "__main__":
   for model_idx in range(0, N_MODELS):
        print(f"####Training model {model_idx}")
        model = deepcopy(model_base)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=1e-12)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.4, total_steps=EPOCHS, anneal_strategy='linear', pct_start=0.15)

        samples_all = np.arange(0, n_cifar)  
        np.random.shuffle(samples_all)
        
        print("##### Training the early-stopped version")
        samples_train = samples_all[:TRAIN_SIZE]
        samples_val = samples_all[TRAIN_SIZE:TRAIN_VAL_SIZE]
        samples_test = samples_all[TRAIN_VAL_SIZE:]

        
        if not os.path.exists(f'{model_training_folder}/val_stopped/model_{model_idx}'):
            os.makedirs(f'{model_training_folder}/val_stopped/model_{model_idx}')
        with open("{}/val_stopped/model_{}/model_{}_samples.json".format(model_training_folder, model_idx, model_idx), 'w') as sample_f:
            samples = {"train": samples_train, "val": samples_val, "test": samples_test}

        trainset = torch.utils.data.Subset(cifar100, samples_train)
        trainset = AugmentedDataset(trainset, augment_transform)
        valset = torch.utils.data.Subset(cifar100, samples_val)
        testset = torch.utils.data.Subset(cifar100, samples_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, num_workers=8, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(testset, BATCH_SIZE=BATCH_SIZE, num_workers=8, persistent_workers=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_vloss = float("inf")
        best_epoch = EPOCHS

        model_losses = {}

        for epoch in range(EPOCHS):
                print('EPOCH {}:'.format(epoch + 1))

                model = model.to('cuda')
                model.train(True)
                avg_loss = train_one_epoch(model, optimizer, epoch, None, train_loader, loss_fn)
                model.train(False)

                running_vloss, vacc = evaluate_model(model, val_loader)
                model = model.to('cpu') # free up gpu memory for ripser

                avg_vloss = running_vloss / (i + 1)
                print('LOSS train {} valid {}, valid acc {}'.format(avg_loss, avg_vloss, vacc))

                # Track best performance, and save the model's state
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    best_epoch = epoch
                    model_path = f'{model_training_folder}/val_stopped/model_{model_idx}/model_{model_idx}_best'
                    torch.save(model, model_path)
                if epoch - best_epoch > PATIENCE:
                    print(f"Early stopping on epoch {epoch}, best epoch {best_epoch}")
                    break

                if epoch % SAVE_EVERY == 0:
                    model_path = f'{model_training_folder}/val_stopped/model_{model_idx}/model_{model_idx}_epoch_{epoch}'
                    torch.save(model, model_path)
                model_losses[epoch] = (avg_loss, avg_vloss.item())
                with open(f'{model_training_folder}/val_stopped/model_{model_idx}/model_{model_idx}_loss_data.json', 'w') as f:
                    json.dump(model_losses, f)
                    
                scheduler.step()

        # evaluate on the test set
        model = model.to('cuda')
        loss_val_stopped, accuracy_val_stopped = evaluate_model(model, test_loader)
        model = model.to('cpu')
        print("Final loss and accuracy: ", loss_val_stopped, accuracy_val_stopped)

        print("##### Training the topology-stopped version")
        samples_train = samples_all[:TRAIN_VAL_SIZE]
        samples_test = samples_all[TRAIN_VAL_SIZE:]

        if not os.path.exists(f'{model_training_folder}/topology_stopped/model_{model_idx}'):
            os.makedirs(f'{model_training_folder}/topology_stopped/model_{model_idx}')
        with open("{}/topology_stopped/model_{}/model_{}_samples.json".format(model_training_folder, model_idx, model_idx), 'w') as sample_f:
            samples = {"train": samples_train, "val": samples_val}

        model = deepcopy(model_base)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=1e-12)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.4, total_steps=EPOCHS, anneal_strategy='linear', pct_start=0.15)

        trainset = torch.utils.data.Subset(cifar100, samples_train)
        trainset = AugmentedDataset(trainset, augment_transform)
        testset = torch.utils.data.Subset(cifar100, samples_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(testset, BATCH_SIZE=BATCH_SIZE, num_workers=8, persistent_workers=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        highest_change = 0
        highest_change_epoch = EPOCHS

        model_losses = {}
        top_count_history = []

        for epoch in range(EPOCHS):
                print('EPOCH {}:'.format(epoch + 1))

                model = model.to('cuda')
                model.train(True)
                avg_loss = train_one_epoch(model, optimizer, epoch, None, train_loader, loss_fn)
                model.train(False)

                model = model.to('cpu') # free up gpu memory for ripser

                top_count = compute_topology_count_for_model(model, trainset)
                top_count_history.append(top_count)
                topology_change = 0
                if topology_change > highest_change:
                    highest_change = topology_change
                    highest_change_epoch = epoch
                    model_path = f'{model_training_folder}/topology_stopped/model_{model_idx}/model_{model_idx}_best'
                    torch.save(model, model_path)
                if len(top_count_history) > 1:
                    topology_change = abs(top_count_history[-1]-top_count_history[-2])

                if epoch - highest_change_epoch > PATIENCE:
                    print(f"Early stopping on epoch {epoch}, best epoch {best_epoch}")
                    break

                print('Topology change {}'.format(topology_change))

            
                if epoch % SAVE_EVERY == 0:
                    model_path = f'{model_training_folder}/topology_stopped/model_{model_idx}/model_{model_idx}_epoch_{epoch}'
                    torch.save(model, model_path)
                model_losses[epoch] = (avg_loss, avg_vloss.item())
                with open(f'{model_training_folder}/topology_stopped/model_{model_idx}/model_{model_idx}_loss_data.json', 'w') as f:
                    json.dump(model_losses, f)
                    
                scheduler.step()