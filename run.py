import code.postprocessor as postprocessor
import code.preprocessor as preprocessor
import code.training as training
import code.ce_train as ce_train
import time

if __name__ == "__main__":
    '''
    This is main function. It has 3 top level functions:
    1. Preprocess
        It has following key functionalities:
        a. Preprocessor parses the input file and finds variable information. 
        b. Finds unates.
        c. Generates samples using cryptominisat.
        d. Performs train test split and defines data loaders.
    2. Model Training
        It trains the GCLN model and returns a trained model.
    3. Postprocess
        It prepares the error formula and calls picosat to verify.
    '''

    # 1. Preprocess input data
    start_time = time.time()
    args, training_samples, train_loader, validation_loader, input_size, num_of_outputs,\
        num_of_vars, input_var_idx, output_var_idx, io_dict,\
        Xvar, Yvar, verilogformula, verilog, PosUnate, NegUnate, device, inp_samples = preprocessor.preprocess()

    # 2. Feed samples into GCLN
    model, train_loss, valid_loss, final_accuracy, final_epochs, disagreed_index = training.trainer(
        args, train_loader, validation_loader, input_size, num_of_outputs, device, ce_flag=0, ce_loop=0
    )
    print(disagreed_index)
    uncovered_samples = training_samples[disagreed_index]
    print("run.py ", training_samples[disagreed_index], input_var_idx)
    print(uncovered_samples[:, input_var_idx])
    if uncovered_samples.shape[0] > 0:
        final_rem_formula = ""
        final_rem_inp_formula = ""
        for j in range(uncovered_samples.shape[0]):
            rem_formula = ""
            for i in range(len(input_var_idx)):
                if uncovered_samples[j, input_var_idx[i]] == 0:
                    rem_formula += "~i"+str(input_var_idx[i])+" & "
                else:
                    rem_formula += "i"+str(input_var_idx[i])+" & "
            rem_formula = "~("+rem_formula[:-3]+") "
            rem_inp_formula = rem_formula
            for i in range(len(output_var_idx)):
                if uncovered_samples[j, output_var_idx[i]] == 0:
                    rem_formula += "| zero"
                else:
                    rem_formula += "| one"
            rem_formula = "("+rem_formula+")"
            final_rem_formula += rem_formula + " & "
            final_rem_inp_formula += rem_inp_formula + " & "
        print("FINAL: ", final_rem_formula[:-3],
              "final rem inp:", final_rem_inp_formula[:-3])
    else:
        final_rem_formula = ""
        final_rem_inp_formula = ""
    # 3. Postprocess skolem function from GCLN
    skolem_functions, is_valid, counter_example = postprocessor.postprocess(
        args, model, final_accuracy, final_epochs, train_loss[-1],
        train_loss[0]-train_loss[-1], verilogformula,
        input_size, input_var_idx, num_of_outputs, output_var_idx,
        io_dict, Xvar, Yvar, PosUnate, NegUnate, start_time, final_rem_formula[
            :-3], final_rem_inp_formula[:-3], num_of_ce=0
    )

    # 4. Counter Example Loop
    if args.ce:
        if not is_valid:
            # Counter Example Loop
            print("Starting Counter Example Loop")
            ce_train.ce_train_loop(args, training_samples, counter_example, inp_samples, num_of_vars,
                                   input_size, num_of_outputs, input_var_idx, output_var_idx,
                                   io_dict, Xvar, Yvar, device, is_valid, verilogformula,
                                   PosUnate, NegUnate, start_time)

    print("Skolem Functions: ", skolem_functions)
