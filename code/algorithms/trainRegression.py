import copy
import sys
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


def train_regressor(args, architecture, cnf,
                    train_loader, validation_loader, learning_rate,
                    max_epochs, input_size, num_of_outputs, K,
                    device, current_output, ce_flag, ce_loop
                    ):

    train_loss = []
    valid_loss = []
    accuracy_list = []

    # Set regularizers
    lambda1 = 1e-6

    # Initialize network
    if cnf == 'cnf':
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

    # Loss and Optimizer
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    ce_count = 0
    if ce_flag:
        ce_count += 1

    # Loading from checkpoint
    if args.load_saved_model and ce_flag:
        print("Loading checkpoint...")
        load_checkpoint(torch.load('model.pth.tar'), gcln, optimizer)

    # Train network
    max_epochs = max_epochs+1
    epoch = 1
    last_acc = 0
    while epoch < max_epochs:
        gcln.train()
        optimizer.zero_grad()
        train_epoch_loss = 0
        accuracy = 0
        datalen = 0
        train_size = 0
        output = []
        target = []
        disagreed_index = []
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((tgts.size(0), -1)).to(device)
            tgts = tgts.round()
            inps = inps.to(torch.float)
            outs = gcln(inps).to(device)

            gcln_ = copy.deepcopy(gcln)
            gcln_.cnf_layer_1.layer_or_weights = torch.nn.Parameter(
                gcln_.cnf_layer_1.layer_or_weights.round())
            gcln_.cnf_layer_1.layer_and_weights = torch.nn.Parameter(
                gcln_.cnf_layer_1.layer_and_weights.round())
            gcln_.cnf_layer_2.layer_or_weights = torch.nn.Parameter(
                gcln_.cnf_layer_2.layer_or_weights.round())
            gcln_.cnf_layer_2.layer_and_weights = torch.nn.Parameter(
                gcln_.cnf_layer_2.layer_and_weights.round())
            out_ = gcln_(inps)

            if architecture == 1:
                t_loss = criterion(
                    outs, tgts[:, current_output].unsqueeze(-1))
                train_epoch_loss += t_loss.item()
            elif architecture == 2:
                t_loss = criterion(outs, tgts)
                train_epoch_loss += t_loss.item()
            elif architecture == 3:
                l = []
                for i in range(num_of_outputs):
                    l.append(criterion(outs[:, i], tgts[:, i]))
                t_loss = sum(l)
                train_epoch_loss += t_loss.item()/num_of_outputs
            train_size += outs.shape[0]
            t_loss = t_loss + lambda1 * \
                torch.sum(1-gcln.cnf_layer_1.layer_and_weights)
            t_loss = t_loss + lambda1 * \
                torch.sum(gcln.cnf_layer_1.layer_or_weights)

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            if architecture == 1:
                output.append([abs(e)
                              for e in out_.round().flatten().tolist()])
                target.append(tgts[:, current_output].tolist())
                val = (out_.round().squeeze() == tgts[:, current_output])

                accuracy += (out_.round().squeeze() ==
                             tgts[:, current_output]).sum()
                if val == False:
                    disagreed_index.append(batch_idx)
            elif architecture > 1:
                output.append([abs(e)
                              for e in out_.round().flatten().tolist()])
                target.append(tgts.flatten().tolist())
                accuracy += (out_.round() == tgts).sum()
        train_loss.append(train_epoch_loss)
        output = [item for sublist in output for item in sublist]
        target = [item for sublist in target for item in sublist]

        if architecture > 1:
            total_accuracy = accuracy.item()/(train_size*num_of_outputs)
        elif architecture == 1:
            total_accuracy = accuracy.item()/(train_size)

        accuracy_list.append(total_accuracy)
        # print("Accuracy: ", total_accuracy)

        if args.ce and ce_loop < 1000:
            if total_accuracy < 1:
                max_epochs += 1
        else:
            if total_accuracy < 1:
                max_epochs += 1

        util.store_losses(train_loss, valid_loss, accuracy_list)
        util.plot()
        # if total_accuracy > 0.99:
        #     scheduler.step()

        if epoch % 1 == 0:
            sys.stdout.write('\r')
            # print('Epoch {}, Train Loss {}, Accuracy {}'.format(
            #     epoch, round(t_loss.item(), 4), total_accuracy)
            # )
            sys.stdout.write('Epoch {}, Train Loss {}, Accuracy {}'.format(
                epoch, round(t_loss.item(), 4), (total_accuracy)))
            sys.stdout.flush()
            checkpoint = {'state_dict': gcln.state_dict(
            ), 'optimizer': optimizer.state_dict()}
            # save_checkpoint(checkpoint)

        epoch += 1

    with torch.no_grad():
        gcln_.cnf_layer_1.layer_or_weights.data.clamp_(0.0, 1.0)
        gcln_.cnf_layer_1.layer_and_weights.data.clamp_(0.0, 1.0)

    return gcln_, train_loss, valid_loss, total_accuracy, epoch-1, disagreed_index
