from code.algorithms.trainClassification import train_classifier
from code.algorithms.trainRegression import train_regressor

import torch
import torch.nn as nn


'''
	Select Problem:
	0: Regression
	1: Classification
'''


def train(args, architecture, cnf, P, train, train_loader, validation_loader, learning_rate, epochs,
          input_size, num_of_outputs, K, device, current_output, ce_flag, ce_loop):

    if P == 0:
        if train:
            gcln, train_loss, valid_loss, accuracy, epoch, disagreed_index = train_regressor(args, architecture, cnf,
                                                                                             train_loader, validation_loader, learning_rate, epochs, input_size, num_of_outputs, K, device, current_output, ce_flag, ce_loop)
        else:
            # print("no train")
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

    return gcln, train_loss, valid_loss, accuracy, epoch, disagreed_index


'''
Select Architecture and Train using:
'''


def trainer(args, train_loader, validation_loader, input_size,
            num_of_outputs, device, ce_flag, ce_loop):
    if args.architecture == 1:
        final_accuracy = 0
        final_epochs = 0
        model_list = []
        disagreed_indices = []
        print('Training GCLN model:')
        for i in range(num_of_outputs):
            current_output = i
            # print("Training for the current output: ", current_output)
            gcln, train_loss, valid_loss, accuracy, epochs, disagreed_index = train(args, args.architecture, args.cnf,
                                                                                    args.P, args.train, train_loader, validation_loader, args.learning_rate, args.epochs,
                                                                                    input_size, num_of_outputs, args.K, device, current_output, ce_flag, ce_loop
                                                                                    )
            final_accuracy += accuracy
            final_epochs += epochs
            model_list.append(gcln)
            disagreed_indices.append(disagreed_index)
        # print("disagreed indices: ", disagreed_indices)
        return model_list, train_loss, valid_loss, final_accuracy, final_epochs, disagreed_indices
    elif args.architecture == 2 or args.architecture == 3:
        current_output = 0
        gcln, train_loss, valid_loss, final_accuracy, final_epochs, disagreed_index = train(args, args.architecture, args.cnf,
                                                                                            args.P, args.train, train_loader, validation_loader, args.learning_rate, args.epochs,
                                                                                            input_size, num_of_outputs, args.K, device, current_output, ce_flag, ce_loop
                                                                                            )

        return gcln, train_loss, valid_loss, final_accuracy, final_epochs, disagreed_index
