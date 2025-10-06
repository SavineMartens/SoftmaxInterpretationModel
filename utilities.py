import numpy as np 
import math

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