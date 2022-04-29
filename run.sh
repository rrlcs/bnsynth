#!/bin/sh
python3 run.py \
--th=0.5 \
--P=$1 \
--train=1 \
--tnorm_name=product \
--learning_rate=.01 \
--run_for_all_outputs=1 \
--verilog_spec_location=custom_examples \
--verilog_spec=$2 \
--training_size=10000 \
--epochs=$3 \
--batch_size=1 \
--K=$4 \
--preprocessor=1 \
--postprocessor=1 \
--architecture=1 \
--layers=1 \
--cnf=1 \
--ce=1 \
--load_saved_model=0 \
--output_file=experiments/final_results.csv
