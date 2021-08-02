# Boolean Function Synthesis using GCLN

## Boolean Function Synthesis:

Given an existentially quantified Boolean formula ∃Y F(X, Y ) over the set of variables X and Y , the problem of Boolean functional synthesis is to compute a vector of Boolean functions, denoted by Ψ(X) = <ψ1(X), ψ2(X), . . . , ψ|Y |(X)>,
and referred to as Skolem function vector, such that
∃Y F(X, Y) ≡ F(X, Ψ(X)).
In the context of applications, the sets
X and Y are viewed as inputs and outputs, and the formula
F(X, Y ) is viewed as a functional specification capturing the
relationship between X and Y , while the Skolem function vector Ψ(X) allows
one to determine the value of Y for the given X by evaluating Ψ (https://arxiv.org/pdf/2005.06922.pdf)

## Requirements
- PyTorch 
```
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
- NumPy
```
pip3 install numpy
```

- ArgParse
```
pip3 install argparse
```

## To run without Hyperparameter Search
```git checkout 2dc92eba65de1d61fa6a973239eeccf8d7d3b87a```

## To switch to writeup branch
```git checkout writeup```

## Train GCLN and get Skolem Function:
```
python3 run.py --th=0.5 --P=0 --correlated_sampling=0 --train=1 --no_of_samples=100000 --tnorm_name=product --no_of_input_var=2 --epochs=100 --spec=1 --learning_rate=0.0001
```
- th is threshold for boolean output (0.5)
- P value states which Problem Formulation to run (0, 1, 2):
	- Select Problem:
		- 0: Regression
		- 1: Classification 1
		- 2: Classification 2
		- 2: Classification 3 when ```--correlated_sampling=1```
- train=1 train model (train=0 load saved model)
- no_of_samples states the number of random samples to generate (keep it >= 50000)
- tnorm_name declares the type of tnorm to be used
- no_of_input_vars: input variables in the specification
- spec: selects specification from the set of given specifications

## More information on options:
```
python run.py --help
```

## References:
- Manthan: https://arxiv.org/pdf/2005.06922.pdf
- GCLN: https://arxiv.org/pdf/2003.07959.pdf

## Collaborators:
- Aditya Kanade
- Chiranjib Bhattacharyya
- Deepak D'Souza
- Ravi Raja
- Stanly Samuel