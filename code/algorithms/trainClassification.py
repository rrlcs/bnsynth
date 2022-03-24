import copy
import os
from code.model.gcln import *
from code.utils.utils import util

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


# weight init
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def save_checkpoint(state, filename="model.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gcln, optimizer):
    print("=> Loading Checkpoint")
    gcln.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train_classifier(args, architecture, cnf,
                     train_loader, validation_loader, learning_rate,
                     max_epochs, input_size, num_of_outputs, K,
                     device, current_output, ce_flag, ce_loop):
    train_loss = []
    valid_loss = []
    accuracy_list = []

    lambda1 = 1e-5
    lambda2 = 1e-5

    # Initialize network
    print("No of outputs: ", num_of_outputs, K, input_size)
    if cnf:
        if architecture == 1:
            gcln = GCLN_CNF_Arch1(
                input_size, num_of_outputs, K, device).to(device)
        elif architecture == 2:
            gcln = GCLN_CNF_Arch2(
                input_size, num_of_outputs, K, device).to(device)
        elif architecture == 3:
            gcln = GCLN_CNF_Arch3(
                input_size, num_of_outputs, K, device).to(device)
    else:
        if architecture == 1:
            gcln = GCLN_DNF_Arch1(
                input_size, num_of_outputs, K, device).to(device)
        elif architecture == 2:
            gcln = GCLN_DNF_Arch2(
                input_size, num_of_outputs, K, device).to(device)
        elif architecture == 3:
            gcln = GCLN_DNF_Arch3(
                input_size, num_of_outputs, K, device).to(device)

    print("Network")
    print(gcln)
    # gcln.apply(init_weights)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

    ce_count = 0
    if ce_flag:
        ce_count += 1

    # Loading from checkpoint
    if args.load_saved_model and ce_flag:
        print("Loading checkpoint...")
        load_checkpoint(torch.load('model.pth.tar'), gcln, optimizer)

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
            outs = gcln(inps).to(device)

            gcln_ = copy.deepcopy(gcln)
            gcln_.layer_or_weights = torch.nn.Parameter(
                gcln_.layer_or_weights.round())
            gcln_.layer_and_weights = torch.nn.Parameter(
                gcln_.layer_and_weights.round())
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
                    # print("lossssss: ", l[-1], i, outs[:, i], tgts[:, i])
                t_loss = sum(l)
                # t_loss = criterion(outs, tgts)
                # train_epoch_loss += t_loss.item()
                train_epoch_loss += t_loss.item()/num_of_outputs

            # t_loss = criterion(out, tgts)
            train_size += outs.shape[0]
            t_loss = t_loss + lambda1*torch.sum(1-gcln.layer_and_weights)
            t_loss = t_loss + lambda1*torch.linalg.norm(gcln.layer_or_weights, 1) + \
                lambda2*torch.linalg.norm(gcln.layer_and_weights, 1)
            t_loss = t_loss + lambda1*torch.linalg.norm(gcln.layer_or_weights, 2) + \
                lambda2*torch.linalg.norm(gcln.layer_and_weights, 2)

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            if architecture == 1:
                accuracy += (out_.round() == tgts[:, current_output]).sum()
            elif architecture > 1:
                accuracy += (out_.round() == tgts).sum()
        train_loss.append(train_epoch_loss)

        if architecture > 1:
            total_accuracy = accuracy.item()/(train_size*num_of_outputs)
        elif architecture == 1:
            total_accuracy = accuracy.item()/(train_size*inps.shape[0])

        accuracy_list.append(total_accuracy)
        print("Accuracy: ", total_accuracy)
        print(total_accuracy == 1)

        if args.ce and ce_loop < 100:
            if total_accuracy < 1:
                max_epochs += 1
        else:
            if total_accuracy != 1:
                max_epochs += 1

        print('epoch {}, train loss {}'.format(
            epoch, round(train_epoch_loss, 4))
        )

        # print("Training Loss: ", t_loss.item())
        # print("G1: ", gcln.layer_or_weights.data)
        # print("G2: ", gcln.layer_and_weights.data)
        print("Gradient for G1: ", gcln.layer_or_weights.grad)
        print("Gradient for G2: ", gcln.layer_and_weights.grad)

        util.store_losses(train_loss, valid_loss, accuracy_list)
        util.plot()

        # scheduler.step()
        print("learning rate: ", scheduler.get_last_lr())

        if epoch % 1 == 0:
            print('epoch {}, train loss {}'.format(
                epoch, round(t_loss.item(), 4))
            )
            checkpoint = {'state_dict': gcln.state_dict(
            ), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)

        epoch += 1

    with torch.no_grad():
        gcln_.layer_or_weights.data.clamp_(0.0, 1.0)
        gcln_.layer_and_weights.data.clamp_(0.0, 1.0)

    return gcln_, train_loss, valid_loss, total_accuracy, epoch-1
