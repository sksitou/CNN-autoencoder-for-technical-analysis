#import
import csv
import numpy as np
import pandas as pd
from libs.utils import load_data
VOLUME_CONST = 10000000.0


#read csv
FILE_NAME = '2833.csv'
FILE_NAME_OUT = '2833_out.csv'

load_data(FILE_NAME,['Close','Volume'])
'''
with open(FILE_NAME, 'rt') as file:
	reader = csv.reader(file)
	close = [row[4] for row in reader][1:]
	close = map(float,close)
	file.seek(0)
	volume = [row[5] for row in reader][1:]
	volume = map(float,volume)
'''
df = pd.read_csv(FILE_NAME, usecols=[4,5])
close = df.loc[:,'Close'].tolist()
volume = df.loc[:,'Volume'].tolist()
#print type(close[0])
#print volume[0]


#loop through array
def loop(input):
	for e in input:
		pass
	return 0
def detecter(e):
	if str(e) == '?':
		return True
	return False
#print "Contain missing data in close : {0}".format(detecter(close))
#print "Contain missing data in volume : {0}".format(detecter(volume))
#take average function

'''normalize by percentage return'''
#normalize funtion, percentage
#%return: ((t)-(t-1))/(t-1)
end = len(close)
percent_close = [(close[i]-close[i-1])/close[i-1] for i in range(1,end)]
percent_close.insert(0,0)
end = len(volume)
nor_volume = [e/VOLUME_CONST if e != 0 else 0 for e in volume]
#print nor_volume
#print percent_close

'''normalize by scaling'''
#normalize function, [-1,1]
#%return: 2*(x-m)/(M-m)+1
max, min = max(close), min(close)
norm = lambda x: (2*(x-min)/(max-min))-1
nor_close = map(norm,close)
def normal(a):
	max, min = np.amax(a), np.amin(a)
	norm = lambda x: (2*(x-min)/(max-min))-1
	nor_a = map(norm,a)
	print np.amin(nor_a), np.amax(nor_a)
	print nor_a

#write in new csv
'''
close_np = np.array(close)
volume_np = np.array(volume)
percent_close_np = np.array(percent_close)
nor_volume_np = np.array(nor_volume)
output = np.dstack((close_np,volume_np,percent_close_np,nor_volume_np))
'''
print len(close)
print len(volume)
print len(percent_close)
print len(nor_close)
print len(nor_volume)

output = {'Close': close,
        'Volume': volume,
        'Percentage_close': percent_close,
        'Normalized_close': nor_close,
        'Normalized_volume': nor_volume}
df = pd.DataFrame(output, columns=['Close','Volume','Percentage_close','Normalized_close','Normalized_volume'])
df.to_csv(FILE_NAME_OUT)
