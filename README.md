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
- Z3
```
pip install z3-solver
```
-ANTLR4
```
pip install antlr4-python3-runtime
```

## Switch to updated branch
```git checkout saakha for architecture 3```

## To switch to writeup branch
```git checkout writeup```

## Run individual examples
```
./run.sh 0 sample1.v 10 50 &> log.txt
```
- Arg1: 0 for regression and 1 for classification
- Arg2: Specify file name for verilog specification
- Arg3: Epochs
- Arg4: Number of Clauses

## Run autotest on 6 samples
```python3 scripts/script.py```

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