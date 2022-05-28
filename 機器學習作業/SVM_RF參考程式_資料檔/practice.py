import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')


iris = datasets.load_iris()

X_train,X_test,y_train,y_test =train_test_split(iris['data'],
                                                iris['target'],
                                                test_size=0.25,random_state=0)
print('shape of X_train:{}'.format(X_train.shape))
print('shape of y_train:{}'.format(y_train.shape))
print('='*64)
print('shape of X_test:{}'.format(X_test.shape))
print('shape of y_test:{}'.format(y_test.shape))




