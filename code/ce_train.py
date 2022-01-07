def ce_train_loop(
    training_samples, io_dict, io_dictz3, ret, model,
    num_of_vars, num_of_outputs, input_size,
	start_time, pos_unate, neg_unate, training_size,
	input_var_idx, output_var_idx, P, threshold, batch_size,
	verilog_spec, verilog_spec_location, Xvar, Yvar, 
    verilog_formula, verilog, learning_rate, epochs, 
	K, device, 
	):

    import importlib
    import time
    from code.train import train
    from code.utils import getSkolemFunc as skf
    from code.utils import getSkolemFunc4z3 as skfz3
    from code.utils import plot as pt
    from code.utils.utils import util
    from math import floor
    from typing import OrderedDict

    import numpy as np
    import torch
    from benchmarks import z3ValidityChecker as z3
    from benchmarks.verilog2z3 import preparez3
    from data.dataLoader import dataLoader
    
    loop = 0
    ce_time = 0
    ce_data_time = 0
    n = 5000
    while ret and loop < 50:
        loop += 1
        print("Counter Example Loop: ", loop)
        s = time.time()
        ce = model.repeat(200, 1)
        ce = torch.cat([util.add_noise(ce) for _ in range(20)]).to(torch.double)
        # print("ce shape: ", ce.shape)
        e = time.time()
        data_t = e - s
        ce_data_time += data_t
		# Add counter examples to existing training data
        # print(training_samples.shape, ce.shape, ce)
        training_samples = torch.cat((training_samples, ce))

		# Re-Train only on counter-examples
        # training_samples = ce

        data_size = training_samples.shape[0]
        val_size = floor(data_size*0.2)
        train_size = data_size - val_size
        validation_set = training_samples[train_size:, :]
        training_set = training_samples[:train_size, :]

        print("Training Data: ", training_set.shape, validation_set.shape)

        train_loader = dataLoader(
            training_set, training_size, P,
            input_var_idx, output_var_idx,
            num_of_outputs, threshold, batch_size
        )
        validation_loader = dataLoader(
            validation_set, training_size, P,
            input_var_idx, output_var_idx,
            num_of_outputs, threshold, batch_size
        )

        flag = 0
        # args.epochs += 5
        skf_dict_z3 = {}
        skf_dict_verilog = {}
        for i in range(len(Yvar)):
            current_output = i
            print("Current Output: ", current_output)
            gcln, train_loss, valid_loss = train(
            P, train, train_loader, validation_loader, learning_rate, epochs, 
            input_size, num_of_outputs, current_output, K, device, num_of_vars, 
            input_var_idx, output_var_idx, io_dict, io_dictz3, threshold, 
            verilog_spec, verilog_spec_location, Xvar, Yvar, verilog_formula, 
            verilog, pos_unate, neg_unate
            )
        
            # Extract and Check
            s = time.time()
            skfunc = skfz3.get_skolem_function(
                gcln, num_of_vars,
                input_var_idx, num_of_outputs, output_var_idx, io_dictz3,
                threshold, K
            )
            skf_dict_z3[Yvar[i]] = skfunc[0]

            skfunc = skf.get_skolem_function(
                gcln, num_of_vars,
                input_var_idx, num_of_outputs, output_var_idx, io_dict,
                threshold, K
            )
            skf_dict_verilog[Yvar[i]] = skfunc[0]
            e = time.time()
            # print("Formula Extraction Time: ", e-s)

        # Run the Z3 Validity Checker
        util.store_nn_output(len(skf_dict_z3), list(skf_dict_z3.values()))
        preparez3(verilog_spec, verilog_spec_location, len(skf_dict_z3))
        importlib.reload(z3)
        result, _ = z3.check_validity()
        if result:
            print("Z3: Valid")
        else:
            print("Z3: Not Valid")

        # Write the error formula in verilog
        util.write_error_formula(verilog_spec, verilog, verilog_formula, list(skf_dict_verilog.values()), Xvar, Yvar, pos_unate, neg_unate)

        # sat call to errorformula:
        check, sigma, ret = util.verify(Xvar, Yvar, verilog)
        print("check: {}, ret: {} ".format(check, ret))
        
        if check == 0:
            print("error...ABC network read fail")
            print("Skolem functions not generated")
            print("not solved !!")
            break
        
        if ret == 0:
            print('error formula unsat.. skolem functions generated')
            break
        else:
            print(check, sigma.modelx, sigma.modely, sigma.modelyp, ret)
            ce = torch.from_numpy(
                np.concatenate(
                    (sigma.modelx, sigma.modely)
                    ).reshape((1, num_of_vars))
                ).to(torch.double)

        util.store_losses(train_loss, valid_loss)
        pt.plot()
        # Run the Validity Checker
        # importlib.reload(z3)  # Reload the package
        s = time.time()
        # result, model = z3.check_validity()
        e = time.time()
        ce_time += e-s
        print("Time Elapsed = ", e - start_time)
        n += 1000

    return ret, ce_time, ce_data_time
