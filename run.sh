#!/bin/sh
python3 run.py \
--th=0.5 \
--P=$1 \
--train=1 \
--tnorm_name=product \
--learning_rate=0.5 \
--run_for_all_outputs=1 \
--verilog_spec_location=verilog \
--verilog_spec=$2 \
--training_size=10000 \
--epochs=$3 \
--batch_size=4 \
--K=$4