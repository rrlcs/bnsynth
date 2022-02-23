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
        Xvar, Yvar, verilogformula, verilog, PosUnate, NegUnate, device = preprocessor.preprocess()

    # 2. Feed samples into GCLN
    model, train_loss, valid_loss, final_accuracy, final_epochs = training.trainer(
        args, train_loader, validation_loader, num_of_vars,
        input_size, num_of_outputs, input_var_idx, output_var_idx,
        io_dict, Xvar, Yvar, device, ce_flag=0
    )

    # 3. Postprocess skolem function from GCLN
    skolem_functions, is_valid, counter_example = postprocessor.postprocess(
        args, model, final_accuracy, final_epochs, train_loss[-1],
        train_loss[0]-train_loss[-1], verilogformula,
        input_size, input_var_idx, num_of_outputs, output_var_idx,
        io_dict, Xvar, Yvar, PosUnate, NegUnate, start_time
    )

    # 4. Counter Example Loop
    if args.ce:
        if not is_valid:
            # Counter Example Loop
            print("Starting Counter Example Loop")
            ce_train.ce_train_loop(args, training_samples, counter_example, num_of_vars,
                                   input_size, num_of_outputs, input_var_idx, output_var_idx,
                                   io_dict, Xvar, Yvar, device, is_valid, verilogformula,
                                   PosUnate, NegUnate, start_time)

    print("Skolem Functions: ", skolem_functions)
