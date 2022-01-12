python3 run.py \
--th=0.5 \
--P=$1 \
--train=1 \
--tnorm_name=product \
--learning_rate=0.001 \
--run_for_all_outputs=1 \
--verilog_spec_location=sample_examples \
--verilog_spec=$2 \
--training_size=10000 \
--epochs=$3 \
--batch_size=32 \
--K=$4

# threshold
# 0: Regression; 1: Classification
# 0: load saved model; 1: train model
# select tnorm type from [godel, product]
# set learning rate for the model
# 0: run for single output; 1: run for all outputs
# specify folder name of data
# specify file name for spec.
# number of random samples to generate
# number of epochs to train for
# number of clauses allowed