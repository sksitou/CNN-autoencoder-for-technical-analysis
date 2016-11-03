'''
import matplotlib.pyplot as plt
import datetime
import numpy as np

x = np.array(range(100))
y = np.random.randint(100, size=x.shape)

plt.plot(x,y)
plt.show()
'''

from pandas import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import random

#The following part is just for generating something similar to your dataframe
date1 = "20140605"
date2 = "20140606"

d = {'date': Series([date1]*5 + [date2]*5), 'template': Series(range(5)*2),
'score': Series([random.random() for i in range(10)]) } 

data = DataFrame(d)
#end of dataset generation

fig, ax = plt.subplots()

for temp in range(5):
    dat = data[data['template']==temp]
    dates =  dat['date']
    dates_f = [dt.datetime.strptime(date,'%Y%m%d') for date in dates]
    ax.plot(dates_f, dat['score'], label = "Template: {0}".format(temp))

plt.xlabel("Date")
plt.ylabel("Score")
ax.legend()
plt.show()