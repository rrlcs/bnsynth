import code.preprocessor as preprocessor
import code.training as training
from code.utils.utils import util

if __name__ == "__main__":


    # 1. Preprocess input data

    args, train_loader, validation_loader, input_size, num_of_outputs,\
         num_of_vars, input_var_idx, output_var_idx, io_dict, io_dictz3,\
              Xvar, Yvar, verilogformula, verilog, PosUnate, NegUnate, device = preprocessor.process()
    print(Xvar, Yvar)
    # 2. Feed samples into GCLN

    # skolem_function = model()
    model, train_loss, valid_loss, final_accuracy, final_epochs = training.trainer(args, train_loader, validation_loader, num_of_vars, input_size, 
            num_of_outputs, input_var_idx, output_var_idx, io_dict, io_dictz3, Xvar, Yvar, device)
    
    print("Learned Params: ", model.layer_or_weights, model.layer_and_weights)

    # 3. Postprocess skolem function from GCLN
