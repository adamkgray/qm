import pandas as pd
import numpy as np
from qm import qm
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import ndarray
from pandas import DataFrame

# generate data
x: ndarray = np.arange(-1, 1, 0.01)
noise: ndarray = np.random.uniform(-.4, .5, size=(200,))
y = (2 * x ** 2) + (2 * x) + 2 + noise

# quadratic regression by gradient descent
a, b, c = qm(x, y, epochs=10000)

# plot datapoints
q_data: DataFrame = pd.DataFrame(data={"x": x, "y": y})
sns.scatterplot(data=q_data, x="x", y="y")

# plot regression
q_regression: ndarray = (a * (x ** 2)) + (b * x) + c
plt.plot(x, q_regression, "r-", linewidth=2)

# show
plt.show()
