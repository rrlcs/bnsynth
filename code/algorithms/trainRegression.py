import importlib

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
    max_epochs, input_size, num_of_outputs, K,
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
    from run import store_nn_output
    train_loss = []
    valid_loss = []
    best_loss = float('inf')

    early_stop = 0

    # Set regularizers
    lambda1 = 1e-2
    lambda2 = 1e-2

    # Initialize network
    gcln = GCLN(input_size, num_of_outputs, K, device, P).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(gcln.parameters()), lr=learning_rate)
    if flag:
        load_checkpoint(torch.load('model.pth.tar'), gcln, optimizer)
    
    # Train network
    for epoch in range(1, max_epochs+1):
        gcln.train()
        optimizer.zero_grad()
        train_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(train_loader):
            tgts = tgts.reshape((tgts.size(0), -1)).to(device)
            out = gcln(inps).to(device)
            l = []
            for i in range(num_of_outputs):
                l.append(criterion(out[:, i], tgts[:, i]))
            t_loss = sum(l)

            t_loss = torch.sqrt(criterion(out, tgts))
            t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
                lambda2*torch.linalg.norm(gcln.G2, 1)
            t_loss = t_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
                lambda2*torch.linalg.norm(gcln.G2, 2)

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            train_epoch_loss += t_loss.item()*inps.size(0)
        train_loss.append(train_epoch_loss / len(train_loader.sampler))

        gcln.eval()
        valid_epoch_loss = 0
        for batch_idx, (inps, tgts) in enumerate(validation_loader):
            tgts = tgts.reshape((tgts.size(0), -1)).to(device)
            out = gcln(inps).to(device)
            l = []
            for i in range(num_of_outputs):
                l.append(criterion(out[:, i], tgts[:, i]))
            v_loss = sum(l)

            v_loss = criterion(out, tgts)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 1) + \
                lambda2*torch.linalg.norm(gcln.G2, 1)
            v_loss = v_loss + lambda1*torch.linalg.norm(gcln.G1, 2) + \
                lambda2*torch.linalg.norm(gcln.G2, 2)

            valid_epoch_loss += v_loss.item()*inps.size(0)
        valid_loss.append(valid_epoch_loss / len(validation_loader.sampler))
        
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

        if epoch % 5 == 0:
            print('epoch {}, train loss {}, valid loss {}'.format(
                epoch, round(t_loss.item(), 4), round(v_loss.item(), 4))
            )
            checkpoint = {'state_dict': gcln.state_dict(), 'optimizer':optimizer.state_dict()}
            save_checkpoint(checkpoint)
            
            # Extract and Check
            skfunc = skfz3.get_skolem_function(
                gcln, num_of_vars,
                input_var_idx, num_of_outputs, output_var_idx, io_dictz3,
                threshold, K
            )
            print("train: ", skfunc)
            store_nn_output(num_of_outputs, skfunc)
            # Run the Validity Checker
            # Run the Z3 Validity Checker
            store_nn_output(num_of_outputs, skfunc)
            preparez3(verilog_spec, verilog_spec_location, num_of_outputs)
            importlib.reload(z3)
            result, _ = z3.check_validity()
            if result:
                print("Z3: Valid")
            else:
                print("Z3: Not Valid")
            # sat call to errorformula:
            skfunc = [s.replace("_", "") for s in skfunc]
            skfunc = skf.get_skolem_function(
                gcln, num_of_vars,
                input_var_idx, num_of_outputs, output_var_idx, io_dict,
                threshold, K
            )
            candidateskf = util.prepare_candidateskf(skfunc, Yvar, pos_unate, neg_unate)
            util.create_skolem_function(
                verilog_spec.split('.v')[0], candidateskf, Xvar, Yvar)
            error_content, refine_var_log = util.create_error_formula(
                Xvar, Yvar, verilog_formula)
            util.add_skolem_to_errorformula(error_content, [], verilog)
            check, sigma, ret = util.verify(Xvar, Yvar, verilog)
            print("Result {}, Epoch {}".format(ret==0, epoch))
            if ret == 0:
                return gcln, train_loss, valid_loss

    return gcln, train_loss, valid_loss
