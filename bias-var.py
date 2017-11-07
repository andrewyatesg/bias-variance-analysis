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

    def bias(self, avgH):
        '''

        :param avgH: array of average hypothesis over all X
        :return:
        '''
        return np.average([(avgH[i] - self.target[i]) ** 2 for i in range(0, len(self.x))])

    def var(self, hypoths):
        vars = [np.var(hypoths[i]) for i in range(0, len(self.x))]
        return np.average(vars), vars

    def plot_target(self):
        self.ax.plot(self.x, self.target)

    def plot_simple(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim((-1, 1))
        self.ax.set_ylim((-1.02, 1.02))

        self.plot_target()

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
        var, vari = self.var([[hypoths[j] for j in range(0, self.iters)] for _ in range(0, len(self.x))])
        self.ax.axhspan(average[0] - np.sqrt(var), average[0] + np.sqrt(var), alpha=0.5, color="red")

        print("Bias Simple: ", bias)
        print("Variance Simple: ", var)
        print("Average out-of-sample error: ", bias + var)

    def plot_complex(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim((-1, 1))
        self.ax.set_ylim((-1.5, 1.5))

        self.plot_target()

        hypoths = []
        for i in range(0, self.iters):
            x1, x2 = self.sampleX[i]
            y1, y2 = self.sampleT[i]
            y = ( (y2 - y1) / (x2 - x1) ) * (self.x - x1) + y1

            hypoths.append(y)
            self.ax.plot(self.x, y, "b", linewidth=0.3)

        average = [np.average(np.transpose(hypoths)[i]) for i in range(0, len(self.x))]
        self.ax.plot(self.x, average, "r", linewidth=1.5)

        bias = self.bias(average)
        var, vari = self.var(np.transpose(hypoths))
        self.ax.fill_between(self.x, average - np.sqrt(vari), average + np.sqrt(vari), facecolor="red", alpha=0.5)

        print("Bias Complex: ", bias)
        print("Variance Complex: ", var)
        print("Average out-of-sample error: ", bias + var)


biasvar = BiasVarDecomp(iters=50)
biasvar.plot_simple()
biasvar.plot_complex()

plt.show()