import numpy as np 
import math
import datetime
from decimal import *

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    row = min(a,n//a)
    column = max(a,n//a)
    return row, column

def is_prime(n):
  for i in range(2,n):
    if (n%i) == 0:
      return False
  return True

def find_closest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def Greenwood_function_mm_to_f(mm, max_Ln=35, A = 165.4, alpha = 2.1, k = 0.88):
    if hasattr(mm, "__len__"): # if vector
        f = []
        for m in mm:
            rel_mm = (max_Ln-m)/max_Ln
            f.append(A*(10**(alpha*rel_mm)-k))
    else: # if scalar
        rel_mm = (max_Ln-mm)/max_Ln
        f = A*(10**(alpha*rel_mm)-k)
    return f

def get_time_str(seconds=False):
    if seconds:
        now = datetime.datetime.now()
        sec = float("%d.%d" % (now.second, now.microsecond)) 
        sec = float(Decimal(sec).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        sec = 'm' + str(sec) + 's'
        now = now.replace(second=0, microsecond=0) 
    else:
        now = datetime.datetime.now().replace(second=0, microsecond=0)
        sec = ''
    now = str(now).replace(':00','')
    now = now.replace(':','h')
    str_now = now.replace(' ', '_')
    str_now += sec
    return str_now

from scipy.optimize import curve_fit
def sigmoid(x, L ,x0, k, b):
    # L is responsible for scaling the output range from [0,1] to [0,L]
    # b adds bias to the output and changes its range from [0,L] to [b,L+b]
    # k is responsible for scaling the input, which remains in (-inf,inf)
    # x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    b = max(33, b)
    L = min(100-b, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def fit_sigmoid(xdata, ydata):
            # L       x0            k  b 
    p0 =    [100-33, np.median(xdata), 1, 33] #[max(ydata), np.median(xdata), 1, min(ydata)] # this is an mandatory initial guess
    # bounds = ((33),(100))
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=1e6)
    y = sigmoid(xdata, *popt)
    return y

def bounded_sigmoid(x, y, x0, k):
    b = max(33, b)
    L = min(100-b, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def fit_bounded_sigmoid(xdata, ydata):
    p0 = [np.median(xdata), 1]
    popt, pcov = curve_fit(bounded_sigmoid, xdata, ydata, p0, method='dogbox', maxfev=1e6)
    y = bounded_sigmoid(xdata, *popt)
    return y