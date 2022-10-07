'''Регуляризованная нелинейная логистическая регрессия'''
# import
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')

# Загрузим данные, первые два столбца - оценки, третий - метки классов
data = np.loadtxt('ex2data2.txt', delimiter=',')
X, y = data[:,:2], data[:,2]

# Посмотрим на первые 5 строк
X[:5], y[:5]

#%%
# Функция для визуализации данных, по осям - оценки, каждому классу соответствует свой цвет
def plotData(x, y, xlabel, ylabel, labelPos, labelNeg):
    
    # разделим классы
    pos = y==1
    neg = y==0

    # построим
    plt.scatter(x[pos, 0], x[pos, 1], s=30, c='darkblue', marker='+', label=labelPos)
    plt.scatter(x[neg, 0], x[neg, 1], s=30, c='yellow', marker='o', edgecolors='b', label=labelNeg)

    # подписи
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x[:, 0].min(), x[:, 0].max())
    plt.ylim(x[:, 1].min(), x[:, 1].max())

    # легенда
    pst = plt.legend(loc='upper right', frameon=True)
    pst.get_frame().set_edgecolor('k');
 #%%   
# Визуализируем данные
plotData(X, y, 'Microchip Test 1', 'Microchip Test 2', 'Accepted', 'Rejected')
 #%% 
# import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

# Создаем модель
poly = PolynomialFeatures(6)

# Создаем новые признаки - многочлены до 6 степени 
X2 = poly.fit_transform(X)
X2.shape 

#%% Определим функции нужные для дальнейшего

# Сигмоида
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Целевая функция J(theta)
def costFunctionR(theta, X, y, lam):
       
    hThetaX = sigmoid(np.dot(X, theta))
    J = - (np.dot(y, np.log(hThetaX)) + np.dot((1 - y), np.log(1 - hThetaX))
    -1/2*lam * np.sum(np.square(theta[1:]))) / len(y)
    return J    


# Приозводная целевой функции J'(theta)
def gradientFuncR(theta, X, y, lam):
    
    hThetaX = sigmoid(np.dot(X, theta))
    
    # Мы не добавляем регуляризацию для θ0, заменим его на 0  
    thetaNoZeroReg = np.insert(theta[1:], 0, 0)
    
    gradient =  (np.dot(X.T, (hThetaX - y)) + lam * thetaNoZeroReg) / len(y) 
    return gradient
#%% 
# Инициализируем нулевой вектор начальных значений 
theta = np.zeros(X2.shape[1])
theta

#%%  Проверка
lam = 1
J = costFunctionR(theta, X2, y, lam)
gradient = gradientFuncR(theta, X2, y, lam)

# cost = 0.693 
print("Cost: %0.3f"%(J))
print("Gradient: {0}".format(gradient))
#%%
# Запускаем метод обучения
import scipy.optimize as opt

result = opt.fmin_tnc(func = costFunctionR, 
                    x0 = theta, fprime = gradientFuncR, 
                    args = (X2, y, lam))

theta_optimized = result[0]


