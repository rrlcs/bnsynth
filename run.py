import code.preprocessor as preprocessor

if __name__ == "__main__":

    # 1. Preprocess input data

    args, train_loader, validation_loader, input_size, num_of_outputs,\
         num_of_vars, input_var_idx, output_var_idx, io_dict, io_dictz3,\
              Xvar, Yvar, verilogformula, verilog, PosUnate, NegUnate, device = preprocessor.process()

    # 2. Feed samples into GCLN

    # skolem_function = model()

    # 3. Postprocess skolem function from GCLN
