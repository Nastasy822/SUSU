
# import
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')

# Загрузим данные, первые два столбца - оценки, третий - метки классов
data = np.loadtxt('ex2data1.txt', delimiter=',')
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
plotData(X, y, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not Admitted')

#%% Определим функции нужные для дальнейшего

# Сигмоида
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Целевая функция J(theta)
def costFunction(theta, X, y):
       
    hThetaX = sigmoid(np.dot(X, theta))
    J = - (np.dot(y, np.log(hThetaX)) + np.dot((1 - y), np.log(1 - hThetaX))) / len(y)
    return J    


# Приозводная целевой функции J'(theta)
def gradientFunc(theta, X, y):
    
    hThetaX = sigmoid(np.dot(X, theta))
    gradient =  np.dot(X.T, (hThetaX - y)) / len(y) 
    return gradient

#%% Проверим их работу

# Добавим столбец из 1
X = np.hstack((np.ones((X.shape[0],1)), X))

# Зададим нулевые значения theta
theta = np.zeros(X.shape[1])
theta

J = costFunction(theta, X, y)
gradient = gradientFunc(theta, X, y)

# Должно быть 0.693 и (-0.1, -12, -11)
print("Cost: %0.3f"%(J))
print("Gradient: {0}".format(gradient))


#%% Минимизируем целевую функцию с помощью метода Ньютона сопряженных градиентов
# реализованного в SciPy 
import scipy.optimize as opt

result = opt.fmin_tnc(func = costFunction, 
                    x0 = theta, fprime = gradientFunc, 
                    args = (X, y))

theta_optimized = result[0]
print(theta_optimized)

#%% Построим границу решения



#%%
# Предположим, студент имеет оценки Exam 1 - 45 и Exam 2 - 85, найдите его
# вероятность поступить. (должно получится 0.776)


#%% Оцените точность классификатора












