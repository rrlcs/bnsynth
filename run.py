import code.postprocessor as postprocessor
import code.preprocessor as preprocessor
import code.training as training
import time
from code.utils.utils import util

if __name__ == "__main__":


    # 1. Preprocess input data
    start_time = time.time()
    args, train_loader, validation_loader, input_size, num_of_outputs,\
         num_of_vars, input_var_idx, output_var_idx, io_dict, io_dictz3,\
              Xvar, Yvar, verilogformula, verilog, PosUnate, NegUnate, device = preprocessor.process()
    # print(Xvar, Yvar)
    # 2. Feed samples into GCLN

    # skolem_function = model()
    model, train_loss, valid_loss, final_accuracy, final_epochs = training.trainer(args, train_loader, validation_loader, num_of_vars, input_size, 
            num_of_outputs, input_var_idx, output_var_idx, io_dict, io_dictz3, Xvar, Yvar, device)
    # print("model list: ", model)
    # if args.architecture==1:
    #     print("Learned Params: ", model[0].layer_or_weights, model[0].layer_and_weights)
    # else:
    #     print("Learned Params: ", model.layer_or_weights, model.layer_and_weights)

    # 3. Postprocess skolem function from GCLN
    skolem_functions, is_valid = postprocessor.postprocess(args, model, final_accuracy, final_epochs, train_loss[-1], train_loss[0]-train_loss[-1], verilogformula,
                                                    input_size, input_var_idx, num_of_outputs, output_var_idx, io_dict, Xvar, Yvar, PosUnate, NegUnate, start_time)
