'''
	Select Problem:
	0: Regression
	1: Classification 1
	2: Classification 2
	3: Classification 3
'''
def train(P, train, train_loader, validation_loader, learning_rate, epochs, 
				input_size, num_of_outputs, current_output, K, device, num_of_vars, input_var_idx,
                output_var_idx, io_dict, io_dictz3, threshold,
                verilog_spec, verilog_spec_location,
                Xvar, Yvar, verilog_formula, verilog, pos_unate, neg_unate):

    from code.algorithms import trainClassification2ndForm as tc2
    from code.algorithms.trainClassification import train_classifier
    from code.algorithms.trainRegression import train_regressor
    from code.model import gcln

    import torch
    import torch.nn as nn
    
    if P == 0:
        if train:
            gcln, train_loss, valid_loss, accuracy, epoch = train_regressor(
                train_loader, validation_loader, learning_rate, epochs, input_size, num_of_outputs, current_output, K, device, num_of_vars, input_var_idx,
                output_var_idx, io_dict, io_dictz3, threshold,
                verilog_spec, verilog_spec_location,
                Xvar, Yvar, verilog_formula, verilog, pos_unate, neg_unate)
        else:
            print("no train")
            gcln = gcln.GCLN(input_size, len(output_var_idx),
                             K, device, P, p=0).to(device)
            gcln.load_state_dict(torch.load("regressor_multi_output"))
            gcln.eval()
            print(list(gcln.G1))
    elif P == 1:
        if train:
            loss_fn = nn.BCEWithLogitsLoss()
            gcln, train_loss, valid_loss = train_classifier(
                train_loader, validation_loader, loss_fn, learning_rate, epochs, input_size, num_of_outputs, K, device, P)
            torch.save(gcln.state_dict(), "classifier1")
        else:
            gcln = gcln.GCLN(input_size, K, device,
                             P).to(device)
            gcln.load_state_dict(torch.load("classifier1"))
            gcln.eval()
    # elif P == 2:
    # 	if train:
    #         loss_fn = nn.BCEWithLogitsLoss()
    #         gcln, train_loss, valid_loss = tc2.train_classifier(
    #             train_loader,
    #             validation_loader, 
    #             loss_fn, 
    #             learning_rate, 
    #             epochs, 
    #             input_size,
    #             num_of_outputs, 
    #             K, 
    #             device, 
    #             P, 
    #             torch, 
    #             gcln.GCLN, 
    #             util, 
    #             py_spec
    #             )
    #         torch.save(gcln.state_dict(), "classifier2")
        # else:
        #     gcln = gcln.GCLN(input_size, K, device, P, p=0).to(device)
        #     gcln.load_state_dict(torch.load("classifier2"))
        #     gcln.eval()
	
    return gcln, train_loss, valid_loss, accuracy, epoch
