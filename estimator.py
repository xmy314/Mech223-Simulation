import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
from sklearn.linear_model import (
    LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
)
from math import ceil,sqrt,floor
import pickle


def generate_data(N=10000):
    
    def f(t):
        return -(0.154/0.15134222365)*((t**2)-(0.913240567396*t)+0.208458255328)/(4.93*(t**2)+t+0.36)

    def fp(t):
        return -(2501391739/2566197470594889259440)*(57439591488032459*(t**2)-13940547232871640*t-5608220395969068)/((493*(t**2)+100*t+36)**2)

    l_s=np.arange(N)

    modulus=ceil(0.5*sqrt(N))
    l_r=0.2*( l_s % modulus)/sqrt(N)-0.02
    l_t=0.4*np.floor(l_s/modulus)/(2*sqrt(N))+0.1
    
    ft=f(l_t)
    fpt=fp(l_t)
    btm=1/np.sqrt(1+(fpt)**2)
    x1=l_t-l_r*fpt*btm
    x2=ft+l_r*btm
    
    return [np.vstack((x1,x2)).T,l_t[:]]

def pre_process(rz):
    return PolynomialFeatures(20).fit_transform(rz)

estimators = [
    LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
]

X, Y = generate_data(1000000)

X = pre_process(X)
test_size=0.2
seed=7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    
model = LinearRegression()
model.fit(X_train,Y_train)


print(model.score(X_test,Y_test))


filename = 'estimator.sav'
pickle.dump(model, open(filename, 'wb'))
 

# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)