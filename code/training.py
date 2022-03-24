# from code.algorithms import trainClassification2ndForm as tc2
from code.algorithms.trainClassification import train_classifier
from code.algorithms.trainRegression import train_regressor

import torch
import torch.nn as nn

# from code.model import gcln


'''
	Select Problem:
	0: Regression
	1: Classification 1
	2: Classification 2
	3: Classification 3
'''


def train(args, architecture, cnf, P, train, train_loader, validation_loader, learning_rate, epochs,
          input_size, num_of_outputs, K, device, current_output, ce_flag, ce_loop):

    if P == 0:
        if train:
            gcln, train_loss, valid_loss, accuracy, epoch = train_regressor(args, architecture, cnf,
                                                                            train_loader, validation_loader, learning_rate, epochs, input_size, num_of_outputs, K, device, current_output, ce_flag, ce_loop)
        else:
            print("no train")
            gcln = gcln.GCLN(input_size, num_of_outputs,
                             K, device, P, p=0).to(device)
            gcln.load_state_dict(torch.load("regressor_multi_output"))
            gcln.eval()
            print(list(gcln.G1))
    elif P == 1:
        if train:
            gcln, train_loss, valid_loss, accuracy, epoch = train_classifier(args, architecture, cnf,
                                                                             train_loader, validation_loader, learning_rate, epochs, input_size, num_of_outputs, K, device, current_output, ce_flag, ce_loop)
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


'''
Select Architecture and Train using:
'''


def trainer(args, train_loader, validation_loader, input_size,
            num_of_outputs, device, ce_flag, ce_loop):
    if args.architecture == 1:
        final_accuracy = 0
        final_epochs = 0
        model_list = []
        for i in range(num_of_outputs):
            current_output = i
            print("Training for the current output: ", current_output)
            gcln, train_loss, valid_loss, accuracy, epochs = train(args, args.architecture, args.cnf,
                                                                   args.P, args.train, train_loader, validation_loader, args.learning_rate, args.epochs,
                                                                   input_size, num_of_outputs, args.K, device, current_output, ce_flag, ce_loop
                                                                   )
            final_accuracy += accuracy
            final_epochs += epochs
            model_list.append(gcln)
        return model_list, train_loss, valid_loss, final_accuracy, final_epochs
    elif args.architecture == 2 or args.architecture == 3:
        current_output = 0
        gcln, train_loss, valid_loss, final_accuracy, final_epochs = train(args, args.architecture, args.cnf,
                                                                           args.P, args.train, train_loader, validation_loader, args.learning_rate, args.epochs,
                                                                           input_size, num_of_outputs, args.K, device, current_output, ce_flag, ce_loop
                                                                           )

        return gcln, train_loss, valid_loss, final_accuracy, final_epochs
