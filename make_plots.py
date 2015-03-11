import matplotlib.pyplot as plt

plt.figure(1)
plt.xlabel('Path Length When Successful', size=20)
plt.ylabel('Frequency', size=20)

# data

# lists of 5 accuracies, one for each class
model1 = []
model2 = []
model3 = []

n, bins, patches = plt.hist([model1, model2, model3], 3, normed=0, label=["Model 1", "Model 2", "Model 3"])    

plt.legend(loc=1)
plt.show()