from imports import plt

f = open("lossess", "r")
lossess = f.read().split(",")
# lossess = lossess.reverse()
lossess = [float(i) for i in lossess]
print(lossess)
plt.plot(lossess)
plt.savefig("lossess.png")