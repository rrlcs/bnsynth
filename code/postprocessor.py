import os
import time
from code.utils.utils import util


def postprocess(args, model, accuracy, epochs, final_loss, loss_drop, verilogformula, num_of_inputs, input_var_idx, num_of_outputs, 
					output_var_idx, io_dict, Xvar, Yvar, PosUnate, NegUnate, start_time):
	
	if args.architecture==1:
		skf_dict = {}
		for i in range(len(model)):
			skolem_function = util.get_skolem_function(args, model[i], num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict)
			skf_dict[Yvar[i]] = skolem_function[0]
		skf_list = list(skf_dict.values())
	elif args.architecture==2:
		skf_list = util.get_skolem_function(args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict)
	else:
		skf_list = util.get_skolem_function(args, model, num_of_inputs, input_var_idx, num_of_outputs, output_var_idx, io_dict)

	if args.postprocessor == 1:
		inputfile_name = args.verilog_spec.split('.v')[0]
		
		# Write the error formula in verilog
		util.write_error_formula1(inputfile_name, args.verilog_spec, verilogformula, skf_list, Xvar, Yvar, PosUnate, NegUnate)

		# sat call to errorformula:
		check, sigma, ret = util.verify(Xvar, Yvar, args.verilog_spec)
		print(check, ret)
		is_valid = 0
		if check == 0:
			print("error...ABC network read fail")
			print("Skolem functions not generated")
			print("not solved !!")
			# exit()
		
		if ret == 0:
			print('error formula unsat.. skolem functions generated')
			print("success")
			is_valid = 1
			skfunc = [sk.replace('\n', '') for sk in skf_list]
			t = time.time() - start_time
			datastring = str(args.verilog_spec)+", "+str(epochs)+", "+str(args.batch_size)+", "+str(args.learning_rate)+", "+str(args.K)+", "+str(len(input_var_idx))+", "+str(num_of_outputs)+", "+str(0)+", "+'; '.join(skfunc)+", "+"Valid"+", "+str(t)+", "+str(final_loss)+", "+str(loss_drop)+", "+str(accuracy)+"\n"
			print(datastring)
			f = open(args.output_file, "a")
			f.write(datastring)
			f.close()

			os.system('rm data/benchmarks/cav20_manthan_dataset/verilog/*.cnf')

	elif args.postprocessor == 2:

		print("skolem functions: ", skf_list)
		verilogformula = util.convert_verilog("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, 0)
		inputfile_name = ("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec).split('/')[-1][:-8]
		verilog = inputfile_name+".v"
		
		# Write the error formula in verilog
		util.write_error_formula2(inputfile_name, verilog, verilogformula, skf_list, Xvar, Yvar, PosUnate, NegUnate)
		
		# sat call to errorformula:
		check, sigma, ret = util.verify(Xvar, Yvar, verilog)
		print(check, ret)
		if check == 0:
			print("error...ABC network read fail")
			print("Skolem functions not generated")
			print("not solved !!")
			is_valid = 0
			# exit()
		
		if ret == 0:
			print('error formula unsat.. skolem functions generated')
			print("success")
			is_valid = 1
			skfunc = [sk.replace('\n', '') for sk in skf_list]
			t = time.time() - start_time
			datastring = str(args.verilog_spec)+", "+str(epochs)+", "+str(args.batch_size)+", "+str(args.learning_rate)+", "+str(args.K)+", "+str(len(input_var_idx))+", "+str(num_of_outputs)+", "+str(0)+", "+'; '.join(skfunc)+", "+"Valid"+", "+str(t)+", "+str(final_loss)+", "+str(loss_drop)+", "+str(accuracy)+"\n"
			print(datastring)
			f = open(args.output_file, "a")
			f.write(datastring)
			f.close()
	
	return skf_list, is_valid
