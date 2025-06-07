import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import acf, pacf

def ts_plot(y, lags=None, title='', fig_size=(14,12), max=50):
    """
    Calcul de l'acf, pacf, de l'histogramme et du QQ-plot d'une série temp
    """
    # on transforme en Series si l'argument y n'en est pas une
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # initialisation de la figure et des axes
    fig = plt.figure(figsize=fig_size)
    ts_ax = fig.add_subplot(221)  # Premier sous-graphe en haut à gauche
    hist_ax = fig.add_subplot(222)  # Deuxième sous-graphe en haut à droite
    acf_ax = fig.add_subplot(223)  # Troisième sous-graphe en bas à gauche
    pacf_ax = fig.add_subplot(224)  # Quatrième sous-graphe en bas à droite
    
    # la serie temporelle
    y.plot(ax=ts_ax)
    ts_ax.set_title(title);
    
    # ACF et PACF
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05, method='ywm')
    
    # histogramme
    y.plot(ax=hist_ax, kind='hist', bins=max);
    hist_ax.set_title('Histogramme');
    plt.tight_layout();
    plt.show()
    

def plot_acf_pacf(y, plot_pacf=False, fig_size=(14,6), lim=20):
    """
    représentation des sorties ACF/PACF
    """
    label = 'ACF'
    y_len = len(y)
    y2 = acf(y.values, fft=True)
    
    if plot_pacf:
        y2 = pacf(y.values)[1:]
        label = 'PACF'
        
    plt.figure(figsize=fig_size)
    plt.bar(range(len(y2)), y2, width = 0.1)
    plt.xlabel('lag')
    plt.ylabel(label)
    plt.axhline(y=0, color='black')
    plt.axhline(y=-1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.axhline(y=1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.ylim(-1, 1)
    plt.xlim(0,lim)
    plt.show()
    
    
def plotseasonal(res, axes ):
    """
    représentation des différentes composantes d'une série
    """
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')

def split_by_year(sunspot, log_sunspot, split_year):
    """
    Splits the sunspot and log_sunspot datasets into training and testing sets
    based on the specified split_year.

    Parameters:
    - sunspot: The sunspot dataset.
    - log_sunspot: The log-transformed sunspot dataset.
    - split_year: The year to split the datasets (training data will be up to this year, testing data will be from the next year onward).

    Returns:
    - x_train, x_test: The training and testing datasets for sunspot.
    - y_train, y_test: The training and testing datasets for log_sunspot.
    """
    x_train = log_sunspot.loc[:str(split_year), 'sunspot']
    x_test = log_sunspot.loc[str(split_year+1):, 'sunspot']

    print(len(x_train))
    print(len(x_test))

    # y_train = log_sunspot.loc[:str(split_year), 'sunspot']
    # y_test = log_sunspot.loc[str(split_year+1):, 'sunspot']
    
    return x_train, x_test#, y_train, y_test
