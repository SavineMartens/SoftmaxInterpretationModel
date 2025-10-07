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