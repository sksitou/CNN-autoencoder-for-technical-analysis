import os
'''
n_pooling_list=['2','3','4']
input_size_list=['1024']
for i in input_size_list:
	for j in n_pooling_list:
		print i,j
		os.system('python run_nn.py {train} {input_size} {n_pooling}'
			.format(train='train',input_size=i,n_pooling=j))
'''
n_filters=['2']
corruption_list = ['0.01','0.05','0.1','0.2','0.5']
'''
graph_list = ['ts.straight\(\)',
				'ts.uptrend\(ts.straight\(\)\)',
				'ts.sinx',
				'ts.uptrend(ts.sinx)']
'''
for k in n_filters:
	for i in corruption_list:
		print k,i
		os.system('python run_nn.py {train} {corruption} {n_filters}'
			.format(train='train',corruption=i,n_filters=k))