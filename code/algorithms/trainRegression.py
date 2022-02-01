import copy
import importlib
from code.train import train
from code.utils import plot as pt

import torch
import torch.nn as nn


def save_checkpoint(state, filename="model.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gcln, optimizer):
    print("=> Loading Checkpoint")
    gcln.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def train_regressor(
    train_loader, validation_loader, learning_rate,
    max_epochs, input_size, num_of_outputs, current_output, K,
    device, P, flag, num_of_vars, input_var_idx,
    output_var_idx, io_dict, io_dictz3, threshold,
    verilog_spec, verilog_spec_location,
    Xvar, Yvar, verilog_formula, verilog, pos_unate, neg_unate
):

    from code.model.gcln import GCLN
    from code.utils import getSkolemFunc as skf
    from code.utils import getSkolemFunc4z3 as skfz3
    from code.utils.utils import util

    from benchmarks import z3ValidityChecker as z3
    from benchmarks.verilog2z3 import preparez3
    train_loss = []
    valid_loss = []
    best_loss = float('inf')

    early_stop = 0

    # Set regularizers
    lambda1 = 1e+2
    lambda2 = 1e-2
    # print("train reg: ", num_of_outputs)

    # Initialize network
    gcln = GCLN(input_size, num_of_outputs, K, device, P).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(list(gcln.parameters()), lr=learning_rate)
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
            # print("tgts:", tgts)
            out = gcln(inps).to(device)
            gcln_ = copy.deepcopy(gcln)
            gcln_.G1 = torch.nn.Parameter(gcln_.G1.round())
            gcln_.G2 = torch.nn.Parameter(gcln_.G2.round())
            # print("model: ", gcln_.G1, gcln_.G2)
            out_ = gcln_(inps)
            train_size += out.shape[0]
            # print("out shape, tgts shape: ", out.shape, tgts.shape)
            l = []
            for i in range(num_of_outputs):
                l.append(criterion(out[:, i], tgts[:, i]))
            t_loss = sum(l)
            # print("loss: ", t_loss)
            # print("target and input shapes: ", out.squeeze().shape, tgts[:, current_output].shape)

            # check network output:
            # print("comparing nw out: ",inps, out, tgts)
            # t_loss = (criterion(out, tgts))
            # print("loss: ", t_loss)
            train_epoch_loss += t_loss.item()/num_of_outputs
            t_loss = t_loss + lambda1*torch.sum(1-gcln.G2)
            # t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
            #     lambda2*torch.linalg.norm(gcln.G2, 1)
            # t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
            #     lambda2*torch.linalg.norm(gcln.G2, 2)
            # print("G1: ", gcln.G1.data)
            # print("G2: ", gcln.G2.data)
            # print("Loss: ", t_loss.item())
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            # print("Gradient for G1: ", gcln.G1.grad)
            # print("Gradient for G2: ", gcln.G2.grad)
            # t_loss = torch.sqrt(criterion(out, tgts[:, current_output].unsqueeze(-1)))
            # print("Loss: ", t_loss.item())
            # print("G1: ", gcln.G1.data)
            # print("G2: ", gcln.G2.data)
            # train_epoch_loss += t_loss.item()*inps.size(0)
            # print(out, out_, tgts)
            accuracy += (out_.round()==tgts).sum()
        total_accuracy = accuracy.item()/(train_size*num_of_outputs)
        print("Accuracy: ", total_accuracy)
        print(total_accuracy==1)
        if total_accuracy != 1:
            max_epochs += 1
        
        train_loss.append(train_epoch_loss)

        print('epoch {}, train loss {}'.format(
                epoch, round(t_loss.item(), 4))
            )
        
        print("Training Loss: ", t_loss.item())
        epoch += 1
        util.store_losses(train_loss, valid_loss)
        pt.plot()
        # if gcln.G1[1,0] > 0.5:
        #     return gcln, train_loss, valid_loss
        # gcln.eval()
        # valid_epoch_loss = 0
        # for batch_idx, (inps, tgts) in enumerate(validation_loader):
        #     tgts = tgts.reshape((tgts.size(0), -1)).to(device)
        #     out = gcln(inps).to(device)
        #     l = []
        #     # for i in range(num_of_outputs):
        #     #     l.append(criterion(out[:, i], tgts[:, i]))
        #     # v_loss = sum(l)

        #     v_loss = torch.sqrt(criterion(out, tgts[:, current_output].unsqueeze(-1)))
        #     v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
        #         lambda2*torch.linalg.norm(gcln.G2, 1)
        #     v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
        #         lambda2*torch.linalg.norm(gcln.G2, 2)

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
                epoch-1, round(t_loss.item(), 4))
            )
            print("Gradient for G1: ", gcln.G1.grad)
            print("Gradient for G2: ", gcln.G2.grad)
            checkpoint = {'state_dict': gcln.state_dict(), 'optimizer':optimizer.state_dict()}
            save_checkpoint(checkpoint)
        
            # Extract and Check
            # skfunc = skfz3.get_skolem_function(
            #     gcln, num_of_vars,
            #     input_var_idx, num_of_outputs, output_var_idx, io_dictz3,
            #     threshold, K
            # )
            # print("z3 skf: ", skfunc)
            # # Run the Validity Checker
            # # Run the Z3 Validity Checker
            # util.store_nn_output(num_of_outputs, skfunc)
            # preparez3(verilog_spec, verilog_spec_location, num_of_outputs)
            # importlib.reload(z3)
            # result, _ = z3.check_validity()
            # if result:
            #     print("Z3: Valid")
            # else:
            #     print("Z3: Not Valid")
            # # sat call to errorformula:
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

    return gcln, train_loss, valid_loss, total_accuracy, epoch-1
