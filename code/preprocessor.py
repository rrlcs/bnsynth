import time
from code.utils.generateSamples_manthan import *
from code.utils.preprocess_manthan import *
from code.utils.utils import util

import torch
from data.dataLoader import dataLoader


def convert_verilog(input,cluster):
        # ng = nx.Graph() # used only if args.multiclass

        with open(input, 'r') as f:
            lines = f.readlines()
        f.close()
        itr = 1
        declare = 'module FORMULA( '
        declare_input = ''
        declare_wire = ''
        assign_wire = ''
        tmp_array = []

        for line in lines:
            line = line.strip(" ")
            if (line == "") or (line == "\n"):
                continue
            if line.startswith("c "):
                continue

            if line.startswith("p "):
                continue


            if line.startswith("a"):
                a_variables = line.strip("a").strip("\n").strip(" ").split(" ")[:-1]
                for avar in a_variables:
                    declare += "%s," %(avar)
                    declare_input += "input %s;\n" %(avar)
                continue

            if line.startswith("e"):
                e_variables = line.strip("e").strip("\n").strip(" ").split(" ")[:-1]
                for evar in e_variables:
                    tmp_array.append(int(evar))
                    declare += "%s," %(evar)
                    declare_input += "input %s;\n" %(evar)
                    # if int(evar) not in list(dg.nodes):
                    #     dg.add_node(int(evar))
                continue

            declare_wire += "wire t_%s;\n" %(itr)
            assign_wire += "assign t_%s = " %(itr)
            itr += 1

            clause_variable = line.strip(" \n").split(" ")[:-1]
            for var in clause_variable:
                if int(var) < 0:
                    assign_wire += "~%s | " %(abs(int(var)))
                else:
                    assign_wire += "%s | " %(abs(int(var)))

            assign_wire = assign_wire.strip("| ")+";\n"
            
            ### if args.multiclass, then add an edge between variables of the clause ###

            # if cluster:
            #     for literal1 in clause_variable:
            #         literal1 = abs(int(literal1))
            #         if literal1 in tmp_array:
            #             if literal1 not in list(ng.nodes):
            #                 ng.add_node(literal1)
            #             for literal2 in clause_variable:
            #                 literal2 = abs(int(literal2))
            #                 if (literal1 != abs(literal2)) and (literal2 in tmp_array):
            #                     if literal2 not in list(ng.nodes):
            #                         ng.add_node(literal2)
            #                     if not ng.has_edge(literal1, literal2):
            #                         ng.add_edge(literal1,literal2)



        count_tempvariable = itr

        declare += "out);\n"
        declare_input += "output out;\n"

        temp_assign = ''
        outstr = ''

        itr = 1
        while itr < count_tempvariable:
            temp_assign += "t_%s & " %(itr)
            if itr % 100 == 0:
                declare_wire += "wire tcount_%s;\n" %(itr)
                assign_wire += "assign tcount_%s = %s;\n" %(itr,temp_assign.strip("& "))
                outstr += "tcount_%s & " %(itr)
                temp_assign = ''
            itr += 1

        if temp_assign != "":
            declare_wire += "wire tcount_%s;\n" %(itr)
            assign_wire += "assign tcount_%s = %s;\n" %(itr,temp_assign.strip("& "))
            outstr += "tcount_%s;\n" %(itr)
        outstr = "assign out = %s" %(outstr)


        verilogformula = declare + declare_input + declare_wire + assign_wire + outstr +"endmodule\n"

        return verilogformula

def process():
    # Get Argument Parser
    parser = util.make_arg_parser()
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.preprocessor == 1:
        #Manthan 1 code
        verilog, output_varlist, total_vars, total_varsz3,\
         verilogformula, pos_unate, neg_unate, Xvar,\
              Yvar, Xvar_map, Yvar_map = util.preprocess_wrapper(
                  args.verilog_spec, args.verilog_spec_location
                  )
        
        # result = util.check_unates(pos_unate, neg_unate, Xvar, Yvar, args.verilog_spec[:-2])
        # if result:
        #     unate_data = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All Unates"+"\n"
        #     f = open("unates.csv", "a")
        #     f.write(unate_data)
        #     f.close()
        #     exit("All Unates!")
        # else:
        #     unate_data = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All not Unates"+"\n"
        #     f = open("unates.csv", "a")
        #     f.write(unate_data)
        #     f.close()
        #     exit("All not Unates!")
        # exit() 

        cnf_content, allvar_map = util.prepare_cnf_content(
            verilog, Xvar, Yvar, Xvar_map, Yvar_map, pos_unate, neg_unate
        )

        # generate sample
        samples = util.generate_samples(
            cnf_content, Xvar, Yvar, Xvar_map, Yvar_map, allvar_map,verilog,
            max_samples=args.training_size
            )
        
        training_samples = util.make_dataset_larger(samples)
        training_samples = torch.from_numpy(samples[:100, :]).to(torch.double)
        print(training_samples.shape)

        # Get train test split
        training_set, validation_set = util.get_train_test_split(training_samples)
        print("Total, Train, and Valid shapes", training_samples.shape,
              training_set.shape, validation_set.shape)

        num_of_vars, num_out_vars, num_of_eqns = util.get_var_counts(Xvar, Yvar, verilog)
        print("No. of vars: {}, No. of output vars: {}, No. of eqns: {}".format(num_of_vars, num_out_vars, num_of_eqns))

        # Prepare input output dictionaries
        io_dict, io_dictz3 = util.prepare_io_dicts(total_vars, total_varsz3=[])

         # Obtain variable indices
        input_var_idx, output_var_idx = util.get_var_indices(num_of_vars, output_varlist, io_dict)
        input_size = 2*len(input_var_idx)
        print("Input size: ", input_size)
        print("Output size: ", len(output_var_idx))

        if args.run_for_all_outputs == 1:
            num_of_outputs = len(output_var_idx)
        else:
            num_of_outputs = 1
        
        # load data
        train_loader = dataLoader(training_set, args.training_size, args.P, input_var_idx,
                                  output_var_idx, num_of_outputs, args.threshold, args.batch_size)
        validation_loader = dataLoader(validation_set, args.training_size, args.P, input_var_idx,
                                       output_var_idx, num_of_outputs, args.threshold, args.batch_size)
    else:
        #Manthan 2 code
        Xvar, Yvar, qdimacs_list = parse("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec)
        print("count X variables", len(Xvar))
        print("count Y variables", len(Yvar))

        all_var = Xvar + Yvar
        total_vars = ["i"+str(v) for v in all_var]
        output_varlist = ["i"+str(v) for v in Yvar]

        inputfile_name = args.verilog_spec[:-8]
        cnffile_name = tempfile.gettempdir()+"/"+inputfile_name+".cnf"

        cnfcontent = convertcnf("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, cnffile_name)
        cnfcontent = cnfcontent.strip("\n")+"\n"

        # finding unates:
        print("preprocessing: finding unates (constant functions)")
        start_t = time.time()
        if len(Yvar) > 0:
            PosUnate, NegUnate = preprocess(cnffile_name)
        else:
            print("too many Y variables, let us proceed with Unique extraction\n")
            PosUnate = []
            NegUnate = []
        end_t = time.time()
        print("preprocessing time:", str(end_t-start_t))

        print("count of positive unates", len(PosUnate))
        print("count of negative unates", len(NegUnate))
        print("positive unates", PosUnate)
        print("negative unates", NegUnate)

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
            info = str(args.verilog_spec)+", "+str(len(Xvar))+", "+str(len(Yvar))+", "+"All Unates"+", "+str(end_time-start_time)+"\n"
            f = open("qdimacsinfo.csv", "a")
            f.write(info)
            f.close()
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

        verilogformula = convert_verilog("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec, 0)
        inputfile_name = ("data/benchmarks/"+args.verilog_spec_location+"/"+args.verilog_spec).split('/')[-1][:-8]
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
                num_samples = 10000
        else:
            num_samples = maxsamples
        
        weighted = 1
        adaptivesample = 0

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

                sampling_weights_y_1 += "w %s 0.5\n" % (yvar)
                sampling_weights_y_0 += "w %s 0.1\n" % (yvar)

            if adaptivesample:
                weighted_sampling_cnf = computeBias(
                    Xvar, Yvar, sampling_cnf, sampling_weights_y_1, sampling_weights_y_0, inputfile_name, Unates, args)
            else:
                weighted_sampling_cnf = sampling_cnf + sampling_weights_y_1
            # print(weighted_sampling_cnf)
            print("generating weighted samples")
            samples = generatesample(
                args, num_samples, weighted_sampling_cnf, inputfile_name, 1)
        else:
            print("generating uniform samples")
            samples = generatesample(
                args, num_samples, sampling_cnf, inputfile_name, 0)

        
        Xvar_tmp = [i-1 for i in Xvar]
        _, indices = np.unique(samples[:, Xvar_tmp], axis=0, return_index=True)
        samples = samples[indices, :]
        print("samples: ", samples.shape)
        print(samples)

        training_samples = torch.from_numpy(samples[:100, :]).to(torch.double)
        print(training_samples.shape)

        training_set, validation_set = util.get_train_test_split(training_samples)
        print("Total, Train, and Valid shapes", training_samples.shape,
            training_set.shape, validation_set.shape)

        num_of_vars, num_out_vars = len(Xvar)+len(Yvar), len(Yvar)

        # Prepare input output dictionaries
        io_dict, io_dictz3 = util.prepare_io_dicts(total_vars, total_varsz3=[])

        # Obtain variable indices
        input_var_idx, output_var_idx = util.get_var_indices(num_of_vars, output_varlist, io_dict)
        input_size = 2*len(input_var_idx)
        print("Input size: ", input_size)
        print("Output size: ", len(output_var_idx))

        if args.run_for_all_outputs == 1:
            num_of_outputs = len(output_var_idx)
        else:
            num_of_outputs = 1

        train_loader = dataLoader(training_set, args.training_size, args.P, input_var_idx,
                              output_var_idx, num_of_outputs, args.threshold, args.batch_size)
        validation_loader = dataLoader(validation_set, args.training_size, args.P, input_var_idx,
                                       output_var_idx, num_of_outputs, args.threshold, args.batch_size)

    
    return args, train_loader, validation_loader, input_size, num_of_outputs,\
         num_of_vars, input_var_idx, output_var_idx, io_dict, io_dictz3, Xvar,\
              Yvar, verilogformula, verilog, PosUnate, NegUnate, device
