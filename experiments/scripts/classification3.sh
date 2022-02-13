for((i=1; i <= 6; i++))
do
	echo $i
	python3 run.py --th=0.5 --P=2 --correlated_sampling=1 --train=1 --no_of_samples=100000 --tnorm_name=product --epochs=100 --spec=$i --learning_rate=0.0001
done