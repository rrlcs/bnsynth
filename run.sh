#!/bin/sh
python3 run.py \
--th=0.5 \
--P=$1 \
--train=1 \
--tnorm_name=product \
--learning_rate=0.5 \
--run_for_all_outputs=1 \
--verilog_spec_location=cav20_manthan_dataset/verilog \
--verilog_spec=$2 \
--training_size=10000 \
--epochs=$3 \
--batch_size=32 \
--K=$4 \
--preprocessor=1 \
--postprocessor=1 \
--architecture=3 \
--cnf=1 \
--output_file=experiments/experiments.csv
