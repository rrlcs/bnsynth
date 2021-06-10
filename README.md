# Boolean Function Synthesis using GCLN

## Train GCLN and get Skolem Function:
```
python run.py --th=0.7 --P=0 --train=1 --no_of_samples=100000
```

## Set arguments using argparse:
```
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", metavar="--th", type=float, default=0.8, help="Enter value between 0.5 <= th <= 1")
parser.add_argument("--no_of_samples", metavar="--n", type=int, default=50000, help="Enter n >= 50000")
parser.add_argument("--no_of_input_var", metavar="--noiv", type=int, default=1, help="Enter value >= 1")
parser.add_argument("--K", type=int, default=10, help="No. of Clauses >= 1")
parser.add_argument("--epochs", type=int, default=50, help="No. of epochs to train")
parser.add_argument("--learning_rate", metavar="--lr", type=float, default=0.01, help="Default 0.01")
parser.add_argument("--batch_size", type=int, default=32, help="Enter batch size")
parser.add_argument("--tnorm_name", type=str, default="product", help="godel/product")
parser.add_argument("--P", type=int, default=0, help="0: Regression, 1: Classification with y as labels, 2: Classification with F out as labels")
parser.add_argument("--train", type=int, default=0, help="True/False; False loads the saved model")
args = parser.parse_args()
```

## P value states which Problem Formulation to run:
Select Problem:
- 0: Regression
- 1: Classification with y as labels
- 2: Classification with output of F as labels

## Collaborators:
- Aditya Kanade
- Chiranjib Bhattacharyya
- Deepak D'Souza
- Ravi Raja
- Stanly Samuel