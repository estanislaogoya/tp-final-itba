# Import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def plotDensityHist(varArr):

    # matplotlib histogram
    #plt.hist(varArr, color = 'blue', edgecolor = 'black',
            #bins = 20)

    # seaborn histogram
    sns.distplot(varArr, hist=True, kde=False,
                bins=20, color = 'blue',
                hist_kws={'edgecolor':'black'})
    # Add labels
    #plt.title('Histogram of Difference')
    #plt.xlabel('Difference vs price')
    #plt.ylabel('# tests')


def plotDensityMultiple(varArr):
    for n in varArr:
        sns.distplot(n, hist=False, rug=True)

    plt.title('Prediction error (Prediction-Actual Price)')
    plt.xlabel('(Prediction-Actual Price)')
    plt.ylabel('# Observations')
    plt.show()
