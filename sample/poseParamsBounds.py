import numpy as np
from numpy import pi,inf

def getBounds():
    out_bounds = {'min': [], 'max': []}
    #                              0     1      2     3    4     5     6      7 
    #     8        9      10    11     12      13      14     15   16   17
    #     18    19  20  21   22     23   24   25 26   27     28     29     30
    out_bounds['min'] = np.array([-pi/6, -pi/3, -inf, -inf, -inf, -inf, -inf, -pi/6,
        -pi/2, -5*pi/6, -pi/6, -pi/6, -pi/6, -5*pi/6, -pi/6, -pi/3, 0, -pi/3,
        -pi/6, -pi, -pi, 0, -pi/3, -pi/6, 0, -pi, 0, -pi/3, -pi/3, -pi/3, -pi/6])
    #                             0     1     2    3    4    5    6     7 
    #     8   9   10    11      12   13  14    15   16     17    18    19  20
    #     21     22     23    24  25    26    27     28   29     30
    out_bounds['max'] = np.array([5*pi/6, pi/3, inf, inf, inf, inf, inf, 3*pi/4,
        pi/6, 0, pi/6, 3*pi/4, pi/2, 0, pi/6, pi/3, pi/3, pi/3, 7*pi/6, 0, pi,
        5*pi/6, pi/3, 7*pi/6, pi, pi, 5*pi/6, pi/3, pi/3, pi/3, pi/6])
    return out_bounds
