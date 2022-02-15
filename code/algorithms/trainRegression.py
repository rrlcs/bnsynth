import copy
import importlib
from code.model.gcln import GCLN_Arch1, GCLN_Arch2, GCLN_Arch3
# from code.training import train
from code.utils import plot as pt
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

def train_regressor(architecture,
        train_loader, validation_loader, learning_rate,
        max_epochs, input_size, num_of_outputs, K,
        device, num_of_vars, input_var_idx,
        output_var_idx, current_output, io_dict, io_dictz3, threshold,
        verilog_spec, verilog_spec_location,
        Xvar, Yvar, verilog_formula, verilog, pos_unate, neg_unate
    ):

    # from benchmarks import z3ValidityChecker as z3
    # from benchmarks.verilog2z3 import preparez3
    train_loss = []
    valid_loss = []
    best_loss = float('inf')

    early_stop = 0

    # Set regularizers
    lambda1 = 1e+1
    lambda2 = 1e-1

    # Initialize network
    print("No of outputs: ", num_of_outputs, K, input_size)
    if architecture==1:
        gcln = GCLN_Arch1(input_size, num_of_outputs, K, device).to(device)
    elif architecture==2:
        gcln = GCLN_Arch2(input_size, num_of_outputs, K, device).to(device)
    elif architecture==3:
        gcln = GCLN_Arch3(input_size, num_of_outputs, K, device).to(device)
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
            # print(inps, tgts[:, 0], tgts[:, 1])
            outs = gcln(inps).to(device)
            # print("outs, tgts: ", outs.shape, tgts.shape)
            # print("outs: ", inps, outs, tgts)
            for name, param in gcln.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
            gcln_ = copy.deepcopy(gcln)
            gcln_.layer_or_weights = torch.nn.Parameter(gcln_.layer_or_weights.round())
            gcln_.layer_and_weights = torch.nn.Parameter(gcln_.layer_and_weights.round())
            out_ = gcln_(inps)
            if architecture == 1:
                t_loss = criterion(outs, tgts[:, current_output])
                train_epoch_loss += t_loss.item()
            elif architecture == 2:
                # l = []
                # for i in range(num_of_outputs):
                #     l.append(criterion(outs[:, i], tgts[:, i]))
                # # print("loss list: ", l)
                # t_loss = sum(l)/num_of_outputs
                # print("tloss: ", t_loss)
                t_loss = criterion(outs, tgts)
                # print(t_loss)
                train_epoch_loss += t_loss.item()
            elif architecture == 3:
                outs = outs.squeeze(-1).T
                out_ = out_.squeeze(-1).T
                l = []
                for i in range(num_of_outputs):
                    l.append(criterion(outs[:, i], tgts[:, i]))
                # print("loss list: ", l)
                t_loss = sum(l)
                train_epoch_loss += t_loss.item()/num_of_outputs
            # print("loss sum: ", t_loss)
            # print("loss: ", t_loss)
            # print(outs.shape, tgts.shape)
            train_size += outs.shape[0]
            # t_loss = (criterion(outs, tgts))
            t_loss = t_loss + lambda1*torch.sum(1-gcln.layer_and_weights)
            t_loss = t_loss + lambda2*torch.sum(1-gcln.layer_or_weights)
            # print("loss: ", t_loss)
            # t_loss = t_loss + lambda1*torch.linalg.norm(gcln.layer_or_weights, 1) + \
            #     lambda2*torch.linalg.norm(gcln.layer_and_weights, 1)
            # print("G1: ", gcln.layer_or_weights.data)
            # print("G2: ", gcln.layer_and_weights.data)
            # print(len(train_loader))

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            # print("Gradient for G1: ", gcln.layer_or_weights.grad)
            # print("Gradient for G2: ", gcln.layer_and_weights.grad)
            # print(inps.size(0))
            
            # print("out_, tgts: ", out_, tgts)
            # print((out_.round()==tgts).sum())
            if architecture == 1:
                accuracy += (out_.round()==tgts[:, current_output]).sum()
            elif architecture > 1:
                accuracy += (out_.round()==tgts).sum()
                # print("acc: ", accuracy)
        # print(len(train_loader.sampler))
        train_loss.append(train_epoch_loss)
        # print(len(train_loader)*32*num_of_outputs, num_of_outputs)
        if architecture > 1:
            total_accuracy = accuracy.item()/(train_size*num_of_outputs)
        elif architecture == 1:
            total_accuracy = accuracy.item()/(train_size)
        print("Accuracy: ", total_accuracy)
        print(total_accuracy==1)
        # print(t_loss)
        if total_accuracy != 1:
            max_epochs += 1

        print('epoch {}, train loss {}'.format(
                epoch, round(t_loss.item(), 4))
            )
        
        # print("Training Loss: ", t_loss.item())
        epoch += 1
        # print("G1: ", gcln.layer_or_weights.data)
        # print("G2: ", gcln.layer_and_weights.data)

        # print("Gradient for G1: ", gcln.layer_or_weights.grad)
        # print("Gradient for G2: ", gcln.layer_and_weights.grad)

        util.store_losses(train_loss, valid_loss)
        pt.plot()
        # if epoch % 10==0:
        #     scheduler.step()
        # gcln.eval()
        # valid_epoch_loss = 0
        # for batch_idx, (inps, tgts) in enumerate(validation_loader):
        #     tgts = tgts.reshape((tgts.size(0), -1)).to(device)
        #     outs = gcln(inps)#.to(device)
        #     print(outs[0].shape, tgts.shape)
        #     l = []
        #     for i in range(num_of_outputs):
        #         l.append(criterion(outs[i], tgts[:, i]))
        #     v_loss = sum(l)

            # v_loss = criterion(out, tgts)
            # v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
            #     lambda2*torch.linalg.norm(gcln.G2, 1)
            # v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
            #     lambda2*torch.linalg.norm(gcln.G2, 2)

        #     valid_epoch_loss += v_loss.item()*inps.size(0)
        # valid_loss.append(valid_epoch_loss / len(validation_loader.sampler))
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # if best_loss - valid_epoch_loss > 0.01:
        #     best_loss = valid_epoch_loss
        #     early_stop = 0
        # else:
        #     early_stop += 1
        #     if early_stop > 5:
        #         print("Early Stopping!!")
                # break

        if epoch % 1 == 0:
            print('epoch {}, train loss {}'.format(
                epoch, round(t_loss.item(), 4))
            )
            # print("Gradient for G1: ", gcln.G1.grad)
            # print("Gradient for G2: ", gcln.G2.grad)
            checkpoint = {'state_dict': gcln.state_dict(), 'optimizer':optimizer.state_dict()}
            save_checkpoint(checkpoint)
        
            # Extract and Check
            # skfunc = skfz3.get_skolem_function(
            #     gcln, num_of_vars,
            #     input_var_idx, num_of_outputs, output_var_idx, io_dictz3,
            #     threshold, K
            # )
            # print("z3 skf: ", skfunc)
            # Run the Validity Checker
            # Run the Z3 Validity Checker
            # util.store_nn_output(num_of_outputs, skfunc)
            # preparez3(verilog_spec, verilog_spec_location, num_of_outputs)
            # importlib.reload(z3)
            # result, _ = z3.check_validity()
            # if result:
            #     print("Z3: Valid")
            # else:
            #     print("Z3: Not Valid")
            # sat call to errorformula:
            # skfunc = skf.get_skolem_function(
            #     gcln, num_of_vars,
            #     input_var_idx, num_of_outputs, output_var_idx, io_dict,
            #     threshold, K
            # )
            # candidateskf = util.prepare_candidateskf(skfunc, Yvar, pos_unate, neg_unate)
            # util.create_skolem_function(
            #     verilog_spec.split('.v')[0], candidateskf, Xvar, Yvar)
            # error_content, refine_var_log = util.create_error_formula(
            #     Xvar, Yvar, verilog_formula)
            # util.add_skolem_to_errorformula(error_content, [], verilog)
            # check, sigma, ret = util.verify(Xvar, Yvar, verilog)
            # print("Result {}, Epoch {}".format(ret==0, epoch))
            # if ret == 0:
            #     return gcln, train_loss, valid_loss

    with torch.no_grad():
        gcln_.layer_or_weights.data.clamp_(0.0, 1.0)
        gcln_.layer_and_weights.data.clamp_(0.0, 1.0)
    return gcln_, train_loss, valid_loss, total_accuracy, epoch-1
