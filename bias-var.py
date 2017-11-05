import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation

class BiasVarDecomp():
    def __init__(self, iters=50):
        self.iters = iters

        self.x = np.linspace(-1, 1)
        self.target = np.sin(np.pi * self.x)

        self.sampleX = [np.array((random.uniform(-1, 1), random.uniform(-1, 1))) for _ in range(0, iters)]
        self.sampleT = [np.sin(np.pi * self.sampleX[i]) for i in range(0, iters)]

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim((-1, 1))
        self.ax.set_ylim((-1.01, 1.01))

    def bias(self, avgH):
        '''

        :param avgH: array of average hypothesis over all X
        :return:
        '''
        biases = []
        for i in range(0, len(self.x)):
            biases.append((avgH[i] - self.target[i]) ** 2)

        return np.average(biases)

    def var(self, hypoths):
        vars = []
        for i in range(0, len(self.x)):
            vars.append(np.var(hypoths[i]))

        return np.average(vars)

    def plot_target(self):
        self.ax.plot(self.x, self.target)

    def plot_simple(self):
        hypoths = []
        for i in range(0, self.iters):
            # The simple learning algorithm
            avg = (self.sampleT[i][1] - self.sampleT[i][0]) / 2

            hypoths.append(avg)
            # Plots each final hypoth.
            self.ax.plot(self.x, avg * np.ones(len(self.x)), "b", linewidth=0.3)

        # Plot the average final hypoth.
        average = np.average(hypoths) * np.ones(len(self.x))
        self.ax.plot(self.x, average, "r", linewidth=1.5)

        bias = self.bias(average)
        var = self.var([[hypoths[j] for j in range(0, self.iters)] for _ in range(0, len(self.x))])
        self.ax.axhspan(average[0] - np.sqrt(var), average[0] + np.sqrt(var), alpha=0.5, color="red")

        print("Bias Simple: ", bias)
        print("Variance Simple: ", var)
        print("Average out-of-sample error: ", bias + var)

biasvar = BiasVarDecomp(iters=200)
biasvar.plot_target()
biasvar.plot_simple()

plt.show()