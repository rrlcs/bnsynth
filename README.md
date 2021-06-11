# Boolean Function Synthesis using GCLN

## Boolean Function Synthesis:
Given an existentially quantified Boolean formula ∃Y F(X, Y ) over the set of variables X and Y , the problem of Boolean functional synthesis is to compute a vector of Boolean functions, denoted by Ψ(X) = <ψ1(X), ψ2(X), . . . , ψ|Y |(X)>,
and referred to as Skolem function vector, such that
∃Y F(X, Y) ≡ F(X, Ψ(X)).
In the context of applications, the sets
X and Y are viewed as inputs and outputs, and the formula
F(X, Y ) is viewed as a functional specification capturing the
relationship between X and Y , while the Skolem function vector Ψ(X) allows
one to determine the value of Y for the given X by evaluating Ψ

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

## Train GCLN and get Skolem Function:
```
python run.py --th=0.7 --P=0 --train=1 --no_of_samples=100000
```
- th is threshold for boolean output (0.5 < th < 1)
- P value states which Problem Formulation to run (0, 1, 2):
	- Select Problem:
		- 0: Regression
		- 1: Classification with y as labels
		- 2: Classification with output of F as labels
- train=1 train model (train=0 load saved model)
- no_of_samples states the number of random samples to generate (keep it >= 50000)

## Check other options and change them using argparse:
```
python run.py --help
```

<!-- ## P value states which Problem Formulation to run:
Select Problem:
- 0: Regression
- 1: Classification with y as labels
- 2: Classification with output of F as labels -->

## Collaborators:
- Aditya Kanade
- Chiranjib Bhattacharyya
- Deepak D'Souza
- Ravi Raja
- Stanly Samuel