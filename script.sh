for((i=1; i <= 6; i++))
do
	echo $i
	python3 run.py --th=0.5 --P=2 --train=1 --no_of_samples=50000 --tnorm_name=product --epochs=100 --spec=$i --learning_rate=0.0001
done