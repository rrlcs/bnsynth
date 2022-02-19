<<<<<<< HEAD
import numpy as np
import wandb
import random
wandb.login()

def train_regressor(train_loader, loss_fn, learning_rate, max_epochs, input_size, K, device, name, P, torch, CLN):

    # Set up your default hyperparameters
    sweep_config = {
    'method': 'random'
    }
    metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

    sweep_config['metric'] = metric
    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
            },
        'epochs': {
            'values': [20, 30, 50]
            },
        'l1': {
            'values': [1e-3, 1e-4, 5e-4, 1e-5, 1e-6, 1e-7]
            },
        'l2': {
            'values': [1e-3, 5e-3, 1e-4, 1e-5, 1e-6]
            },
        'K': {
            'values': [5, 10, 15, 20]
            },
        }
    # hyperparameter = dict(
    #     learning_rate=[0.01, 0.001, 0.0001],
    #     optimizer=["adam", "sgd"],
    #     epochs=[30, 50, 75, 100, 150, 200],
    #     l1 = [1e-3, 1e-4, 5e-4, 1e-5, 1e-6, 1e-7],
    #     l2 = [1e-3, 5e-3, 1e-4, 1e-5, 1e-6],
    #     K = [5, 10, 15, 20, 30, 50]
    #     )
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="bfs1")
    
    # wandb.agent(sweep_id, train, count=5)
    def build_optimizer(network, optimizer, learning_rate):
        if optimizer == "sgd":
            optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate, momentum=0.9)
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(network.parameters(),
                                lr=learning_rate)
        return optimizer

    # Initialize a new wandb run
    with wandb.init(config=sweep_config):
        config = wandb.config
        print(config.parameters['K']['values'])
        lossess = []
        K = random.choice(config.parameters['K']['values'])
        lambda1 = random.choice(config.parameters['l1']['values'])
        lambda2 = random.choice(config.parameters['l2']['values'])
        cln = CLN(input_size, K, device, name, P, p=0).to(device)
        # optimizer = torch.optim.Adam(list(cln.parameters()), lr=learning_rate)
        optimizer = build_optimizer(cln, random.choice(config.parameters['optimizer']['values']), random.choice(config.parameters['learning_rate']['values']))
        criterion = loss_fn
        cln.train()
        optimizer.zero_grad()
        emp_loss = []
        # print(config['epochs']['values'])
        epochs = random.choice(config.parameters['epochs']['values'])
        for epoch in range(epochs):
            total_epoch_loss = 0
            for batch_idx, (inps, tgts) in enumerate(train_loader):
                tgts = tgts.reshape((-1, 1)).to(device)
                out = cln(inps)
                loss = criterion(out, tgts)
                loss = loss + lambda1*torch.linalg.norm(cln.G1, 1) + lambda2*torch.linalg.norm(cln.G2, 1)
                if epoch >= max_epochs // 2:
                    emp_loss.append(loss)
                total_epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"batch loss": loss.item()})
            # lossess.append(total_epoch_loss/training_size)
            avg_loss = total_epoch_loss / len(train_loader)
            lossess.append(total_epoch_loss.item() / len(train_loader.dataset))
            wandb.log({"loss": avg_loss, "epoch": epoch, "K":K, "l1":lambda1, "l2":lambda2})
            print("total epoch loss: ", total_epoch_loss)
            if epoch % 5 == 0:
                print('epoch {}, loss {}'.format(epoch, loss.item()))
                # print('cln or weights grad:', cln.G1.grad.data.cpu().numpy().flatten().round(2))
        wandb.agent(sweep_id, train_regressor(train_loader, loss_fn, learning_rate, max_epochs, input_size, K, device, name, P, torch, CLN), count=5)
    emp_loss = np.array([e.detach().numpy() for e in emp_loss])
    if emp_loss.all() < 1e-5:
        print("Zero Error")
    print(len(emp_loss))
    print(len(emp_loss[emp_loss < 0.1]))
    # print("P(loss < 0.1) = ", len(emp_loss[emp_loss < 0.1]) / len(emp_loss))
    return cln, lossess, K, sweep_id
    
=======
import copy
from code.model.gcln import *
from code.utils.utils import util

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


def save_checkpoint(state, filename="model.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gcln, optimizer):
    print("=> Loading Checkpoint")
    gcln.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def train_regressor(architecture, cnf,
        train_loader, validation_loader, learning_rate,
        max_epochs, input_size, num_of_outputs, K,
        device, num_of_vars, input_var_idx,
        output_var_idx, current_output, io_dict, threshold,
        verilog_spec, verilog_spec_location,
        Xvar, Yvar, verilog_formula, verilog, pos_unate, neg_unate
    ):

    train_loss = []
    valid_loss = []

    early_stop = 0

    # Set regularizers
    lambda1 = 1e+1
    lambda2 = 1e+1

    # Initialize network
    print("No of outputs: ", num_of_outputs, K, input_size)
    if cnf:
        if architecture==1:
            gcln = GCLN_CNF_Arch1(input_size, num_of_outputs, K, device).to(device)
        elif architecture==2:
            gcln = GCLN_CNF_Arch2(input_size, num_of_outputs, K, device).to(device)
        elif architecture==3:
            gcln = GCLN_CNF_Arch3(input_size, num_of_outputs, K, device).to(device)
    else:
        if architecture==1:
            gcln = GCLN_DNF_Arch1(input_size, num_of_outputs, K, device).to(device)
        elif architecture==2:
            gcln = GCLN_DNF_Arch2(input_size, num_of_outputs, K, device).to(device)
        elif architecture==3:
            gcln = GCLN_DNF_Arch3(input_size, num_of_outputs, K, device).to(device)
    
    print("Network")
    print(gcln)

    # Loss and Optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(list(gcln.parameters()), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # if flag:
    #     load_checkpoint(torch.load('model.pth.tar'), gcln, optimizer)
    
    # Train network
    max_epochs = max_epochs+1
    epoch = 1
    while epoch < max_epochs:
        gcln.train()
        optimizer.zero_grad()
        train_epoch_loss = 0
        accuracy = 0
        datalen = 0
        train_size = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((tgts.size(0), -1)).to(device)
            tgts = tgts.round()
            outs = gcln(inps).to(device)
            # for name, param in gcln.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            gcln_ = copy.deepcopy(gcln)
            gcln_.layer_or_weights = torch.nn.Parameter(gcln_.layer_or_weights.round())
            gcln_.layer_and_weights = torch.nn.Parameter(gcln_.layer_and_weights.round())
            out_ = gcln_(inps)
            if architecture == 1:
                t_loss = criterion(outs, tgts[:, current_output])
                train_epoch_loss += t_loss.item()
            elif architecture == 2:
                t_loss = criterion(outs, tgts)
                train_epoch_loss += t_loss.item()
            elif architecture == 3:
                outs = outs.squeeze(-1).T
                out_ = out_.squeeze(-1).T
                l = []
                for i in range(num_of_outputs):
                    l.append(criterion(outs[:, i], tgts[:, i]))
                t_loss = sum(l)
                train_epoch_loss += t_loss.item()/num_of_outputs
            train_size += outs.shape[0]
            t_loss = t_loss + lambda1*torch.sum(1-gcln.layer_and_weights)
            # t_loss = t_loss + lambda2*torch.sum(1-gcln.layer_or_weights)
            # t_loss = t_loss + lambda1*torch.linalg.norm(gcln.layer_or_weights, 1) + \
            #     lambda2*torch.linalg.norm(gcln.layer_and_weights, 1)

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            
            if architecture == 1:
                accuracy += (out_.round()==tgts[:, current_output]).sum()
            elif architecture > 1:
                accuracy += (out_.round()==tgts).sum()
        train_loss.append(train_epoch_loss)

        if architecture > 1:
            total_accuracy = accuracy.item()/(train_size*num_of_outputs)
        elif architecture == 1:
            total_accuracy = accuracy.item()/(train_size)
        
        print("Accuracy: ", total_accuracy)
        print(total_accuracy==1)

        if total_accuracy != 1:
            max_epochs += 1

        print('epoch {}, train loss {}'.format(
                epoch, round(t_loss.item(), 4))
            )
        
        # print("Training Loss: ", t_loss.item())
        # print("G1: ", gcln.layer_or_weights.data)
        # print("G2: ", gcln.layer_and_weights.data)
        # print("Gradient for G1: ", gcln.layer_or_weights.grad)
        # print("Gradient for G2: ", gcln.layer_and_weights.grad)

        util.store_losses(train_loss, valid_loss)
        util.plot()

        # if epoch % 1==0:
        #     scheduler.step()        

        if epoch % 1 == 0:
            print('epoch {}, train loss {}'.format(
                epoch, round(t_loss.item(), 4))
            )
            checkpoint = {'state_dict': gcln.state_dict(), 'optimizer':optimizer.state_dict()}
            save_checkpoint(checkpoint)
        
        epoch += 1

    with torch.no_grad():
        gcln_.layer_or_weights.data.clamp_(0.0, 1.0)
        gcln_.layer_and_weights.data.clamp_(0.0, 1.0)

    return gcln_, train_loss, valid_loss, total_accuracy, epoch-1
>>>>>>> saakha
