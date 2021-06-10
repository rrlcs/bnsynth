from imports import plt

f = open("lossess", "r")
lossess = f.read().split(",")
lossess = [float(i) for i in lossess]
plt.plot(lossess)
plt.savefig("lossess.png")