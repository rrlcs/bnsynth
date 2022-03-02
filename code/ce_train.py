import torch
import numpy as np
from code.utils.utils import util
from data.dataLoader import dataLoader
import code.postprocessor as postprocessor
import code.preprocessor as preprocessor
import code.training as training
import time


def ce_train_loop(
    args, training_samples, counter_example, inp_samples, num_of_vars,
    input_size, num_of_outputs, input_var_idx, output_var_idx,
    io_dict, Xvar, Yvar, device, is_valid, verilogformula, PosUnate,
    NegUnate, start_time
):

    loop = 0
    while not is_valid and loop < 200:
        loop += 1
        counter_example = counter_example.numpy()
        ce_inp_sample = tuple(counter_example[:, input_var_idx][0])
        counter_example = torch.tensor(counter_example)
        print("ce inp smaple", ce_inp_sample)
        if ce_inp_sample in inp_samples:
            training_samples = training_samples
        else:
            # counter_example = util.make_dataset_larger(counter_example)
            training_samples = torch.cat(
                (training_samples, counter_example))

        inp_samples = list(training_samples[:, input_var_idx].numpy())
        inp_samples = list(set([tuple(x) for x in inp_samples]))
        # samples = np.array(counter_example)
        # x_data, indices = np.unique(
        #     samples[:, Xvar], axis=0, return_index=True)
        # samples = samples[indices, :]
        print(training_samples.shape)
        training_set, validation_set = util.get_train_test_split(
            training_samples)
        train_loader = dataLoader(training_set, args.training_size, args.P, input_var_idx,
                                  output_var_idx, num_of_outputs, args.threshold, args.batch_size)
        validation_loader = dataLoader(validation_set, args.training_size, args.P, input_var_idx,
                                       output_var_idx, num_of_outputs, args.threshold, args.batch_size)

        # 2. Feed samples into GCLN
        model, train_loss, valid_loss, final_accuracy, final_epochs = training.trainer(
            args, train_loader, validation_loader, num_of_vars,
            input_size, num_of_outputs, input_var_idx, output_var_idx,
            io_dict, Xvar, Yvar, device, ce_flag=1, ce_loop=loop
        )

        # 3. Postprocess skolem function from GCLN
        skolem_functions, is_valid, counter_example = postprocessor.postprocess(
            args, model, final_accuracy, final_epochs, train_loss[-1],
            train_loss[0]-train_loss[-1], verilogformula,
            input_size, input_var_idx, num_of_outputs, output_var_idx,
            io_dict, Xvar, Yvar, PosUnate, NegUnate, start_time
        )

        print("counter example learned skf: ", skolem_functions)

        if is_valid:
            print("skolem function generated succesflly")
            print("Time: ", time.time() - start_time)
