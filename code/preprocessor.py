import tempfile
import time
from code.utils.utils import util

import numpy as np
import torch
from art import *


def preprocess():
    '''
    Generates samples using Manthan's sampler.
    Removes don't cares, if any.
    Further obtains required information from the specification F(X, Y)
    '''

    # Get Argument Parser
    parser = util.make_arg_parser()
    args = parser.parse_args()

    a = text2art("BNSYNTH")
    print(a)
    print('____A Bounded Boolean Functional Synthesis Tool using GCLN____\n')
    print('Benchmark: ', args.verilog_spec)
    print('Bound on Clauses (K): ', args.K)
    print('Format: ', args.cnf.upper())
    print('Architecture: ', args.architecture)
    # input()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    f = open("cnf", "w")
    f.write(str(args.cnf))
    f.close()

    if args.preprocessor == 1:
        # Manthan 1 code
        # print("Starting Manthan1 Preprocessor")
        verilog, output_varlist, total_vars, total_varsz3,\
            verilogformula, PosUnate, NegUnate, Xvar,\
            Yvar, Xvar_map, Yvar_map = util.preprocess_wrapper(
                args.verilog_spec, args.verilog_spec_location
            )

        result = util.check_unates(
            PosUnate, NegUnate, Xvar, Yvar, args.verilog_spec[:-2])
        # if result:
        #     unate_data = str(args.verilog_spec)+", "+str(len(Xvar)) + \
        #         ", "+str(len(Yvar))+", "+"All Unates"+"\n"
        #     f = open("experiments/unates.csv", "a")
        #     f.write(unate_data)
        #     f.close()
        #     exit("All Unates!")
        # else:
        #     unate_data = str(args.verilog_spec)+", "+str(len(Xvar)) + \
        #         ", "+str(len(Yvar))+", "+"All not Unates"+"\n"
        #     f = open("experiments/unates.csv", "a")
        #     f.write(unate_data)
        #     f.close()
        #     exit("All not Unates!")
        # exit()

        cnf_content, allvar_map = util.prepare_cnf_content(
            verilog, Xvar, Yvar, Xvar_map, Yvar_map, PosUnate, NegUnate
        )
        # exit()
        # generate sample
        samples = util.generate_samples(
            cnf_content, Xvar, Yvar, Xvar_map, Yvar_map, allvar_map, verilog,
            max_samples=1000
        )

        num_of_vars, num_out_vars, num_of_eqns = util.get_var_counts(
            Xvar, Yvar, verilog)
        # print("No. of vars: {}, No. of output vars: {}, No. of eqns: {}".format(
        #     num_of_vars, num_out_vars, num_of_eqns))

        # Prepare input output dictionaries
        io_dict = util.prepare_io_dicts(total_vars)
        io_dictz3 = util.prepare_io_dicts(total_varsz3)

        # Obtain variable indices
        input_var_idx, output_var_idx = util.get_var_indices(
            num_of_vars, output_varlist, io_dict)
        input_size = 2*len(input_var_idx)
        inp_samples_list = samples[:, input_var_idx]
        inp_samples_list = [tuple(x) for x in inp_samples_list]
        inp_samples = list(set(inp_samples_list))
        out_samples_list = samples[:, output_var_idx]
        out_samples_list = [tuple(x) for x in out_samples_list]
        out_samples = list(set(out_samples_list))

        # Create dictionary with input sequence as key and output sequence as values
        d = {}
        for i in range(len(out_samples_list)):
            if inp_samples_list[i] in d.keys():
                d[inp_samples_list[i]].append(out_samples_list[i])
            else:
                d[inp_samples_list[i]] = [out_samples_list[i]]

        # Find indices for don't cares
        count = 2**(num_out_vars)
        inds = []
        for k in d.keys():
            if len(d[k]) == count:
                inds.append([i for i, x in enumerate(
                    inp_samples_list) if x == k])
                # print("indices: ", inds)
        total_indices = [i for i in range(len(out_samples_list))]
        inds = [item for sublist in inds for item in sublist]

        # Find indices of samples to keep
        remainder_indices = list(set(total_indices) - set(inds))

        # Filtered don't cares from the data
        samples = samples[remainder_indices, :]
        x_data, indices = np.unique(
            samples[:, Xvar], axis=0, return_index=True)
        samples = samples[indices, :]
        np.random.RandomState(42)
        # if samples.shape[0] > 100:
        #     samples = samples[np.random.choice(
        #         samples.shape[0], 100, replace=False), :]
        # samples = np.random.rand(samples.shape[0], samples.shape[1])
        # samples = samples[:1, :]

        # np.savetxt("samples.csv", samples, delimiter=",")
        # training_samples = util.make_dataset_larger(samples)
        training_samples = torch.from_numpy(samples).to(torch.double)

        # Get train test split
        training_set, validation_set = util.get_train_test_split(
            training_samples)
        # print("Total, Train, and Valid shapes", training_samples.shape,
        #       training_set.shape, validation_set.shape)

        num_of_outputs = len(output_var_idx)

        # load data
        train_loader = util.dataLoader(training_set, input_var_idx,
                                       output_var_idx, args.batch_size)
        validation_loader = util.dataLoader(validation_set, input_var_idx,
                                            output_var_idx, args.batch_size)
    else:
        # Manthan 2 code
        Xvar, Yvar, qdimacs_list = util.parse(
            "benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec)

        all_var = Xvar + Yvar
        total_vars = ["i"+str(v) for v in all_var]
        output_varlist = ["i"+str(v) for v in Yvar]

        inputfile_name = args.verilog_spec[:-8]
        cnffile_name = tempfile.gettempdir()+"/"+inputfile_name+".cnf"

        cnfcontent = util.convertcnf(
            "benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, cnffile_name)
        cnfcontent = cnfcontent.strip("\n")+"\n"

        start_t = time.time()
        if len(Yvar) > 0:
            PosUnate, NegUnate = util.preprocess(cnffile_name)
        else:
            print("too many Y variables, let us proceed with Unique extraction\n")
            PosUnate = []
            NegUnate = []
        end_t = time.time()

        Unates = PosUnate + NegUnate

        for yvar in PosUnate:
            qdimacs_list.append([yvar])
            cnfcontent += "%s 0\n" % (yvar)

        for yvar in NegUnate:
            qdimacs_list.append([-1 * int(yvar)])
            cnfcontent += "-%s 0\n" % (yvar)

        end_time = time.time()
        if len(Unates) == len(Yvar):
            print(PosUnate)
            print(NegUnate)
            print("all Y variables are unates and have constant functions")
            # info = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All Unates"+", "+str(end_time-start_time)+"\n"
            # f = open("qdimacsinfo.csv", "a")
            # f.write(info)
            # f.close()
            # exit()
            # skolemfunction_preprocess(
            #     Xvar, Yvar, PosUnate, NegUnate, [], '', inputfile_name)

            # logtime(inputfile_name, "totaltime:"+str(end_time-start_time))
            # exit()
        # print("Preprocessing Time: ", end_time-start_time)

        # Logging
        # info = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"Not All Unates"+", "+str(end_time-start_time)+"\n"
        # f = open("qdimacsinfo.csv", "a")
        # f.write(info)
        # f.close()

        verilogformula = util.convert_verilog(
            "benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, 0)
        inputfile_name = ("benchmarks/"+args.verilog_spec_location +
                          "/"+args.verilog_spec).split('/')[-1][:-8]
        verilog = inputfile_name+".v"

        # sampling
        maxsamples = 0
        sampling_cnf = cnfcontent
        if not maxsamples:
            if len(Xvar) > 4000:
                num_samples = 1000
            if (len(Xvar) > 1200) and (len(Xvar) <= 4000):
                num_samples = 5000
            if len(Xvar) <= 1200:
                num_samples = 1000
        else:
            num_samples = maxsamples

        weighted = 1
        adaptivesample = 1

        if weighted:
            sampling_weights_y_1 = ''
            sampling_weights_y_0 = ''
            for xvar in Xvar:
                sampling_cnf += "w %s 0.5\n" % (xvar)
            for yvar in Yvar:
                # if yvar in UniqueVars:
                #     sampling_cnf += "w %s 0.5\n" % (yvar)
                #     continue
                if (yvar in PosUnate) or (yvar in NegUnate):
                    continue

                sampling_weights_y_1 += "w %s 0.9\n" % (yvar)
                sampling_weights_y_0 += "w %s 0.1\n" % (yvar)

            if adaptivesample:
                weighted_sampling_cnf = util.computeBias(
                    Xvar, Yvar, sampling_cnf, sampling_weights_y_1, sampling_weights_y_0, inputfile_name, Unates, args)
            else:
                weighted_sampling_cnf = sampling_cnf + sampling_weights_y_1
            samples = util.generatesample(
                args, num_samples, weighted_sampling_cnf, inputfile_name, weighted)
        else:
            samples = util.generatesample(
                args, num_samples, sampling_cnf, inputfile_name, weighted)

        # print("all samples: ", (samples))

        Xvar_tmp = [i-1 for i in Xvar]
        _, indices = np.unique(samples[:, Xvar_tmp], axis=0, return_index=True)
        samples = samples[indices, :]

        training_samples = torch.from_numpy(samples[:100, :]).to(torch.double)

        training_set, validation_set = util.get_train_test_split(
            training_samples)

        num_of_vars, num_out_vars = len(Xvar)+len(Yvar), len(Yvar)

        # Prepare input output dictionaries
        io_dict = util.prepare_io_dicts(total_vars)

        # Obtain variable indices
        input_var_idx, output_var_idx = util.get_var_indices(
            num_of_vars, output_varlist, io_dict)
        input_size = 2*len(input_var_idx)

        num_of_outputs = len(output_var_idx)

        train_loader = util.dataLoader(training_set, input_var_idx,
                                       output_var_idx, args.batch_size)
        validation_loader = util.dataLoader(validation_set, input_var_idx,
                                            output_var_idx, args.batch_size)

    return args, training_samples, train_loader, validation_loader, input_size, num_of_outputs,\
        num_of_vars, input_var_idx, output_var_idx, io_dict, io_dictz3, Xvar,\
        Yvar, verilogformula, verilog, PosUnate, NegUnate, device, inp_samples, total_varsz3
