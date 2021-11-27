import matplotlib.pyplot as plt
def plot():
	f = open("train_loss", "r")
	train_loss = f.read().split(",")
	f.close()
	train_loss = [float(i) for i in train_loss]
	plt.plot(train_loss, label="train_loss")
	f = open("valid_loss", "r")
	valid_loss = f.read().split(",")
	f.close()
	valid_loss = [float(i) for i in valid_loss]
	plt.plot(valid_loss, label="valid_loss")
	plt.legend()
	plt.savefig("train_valid_loss_plot.png")