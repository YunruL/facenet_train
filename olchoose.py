import sqlite3
import sys
import numpy as np
caffe_root = '/home/liuyunr/hha_pair/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.insert(0, caffe_root + 'python/caffe/proto')
import caffe_pb2
import time
from PIL import Image

caffe.set_device(7)
caffe.set_mode_gpu()

alpha = 0.15
beta = 0.1
l1 = 0.89
l2 = 1
conn = sqlite3.connect('oltriple.db')
conn.isolation_level = None
data_root = '/data/caffePjs/ir/ir_face/'

fname = open('/data/caffePjs/ir/ir_list.txt','r')
namelist = [x.strip() for x in fname.readlines()]
fnum = open('/data/caffePjs/ir/ir_num.txt','r')
numline = fnum.readline().strip()
numper = []
amount = 0
while numline!='':
	num = int(numline)
	numper.append({'begin':amount,'num':num})
	amount = amount+num
	numline = fnum.readline().strip()
fnum.close()
pcount = len(numper)
tcount = pcount-500 #use the last 500 people as validation
#land_model: modelname, beginind
#get the newest model
cu = conn.cursor()
itercount = 0
while itercount <100:
	print itercount
	cu.execute('SELECT modelname,beginind FROM ir_model where id = 0')
	result = cu.fetchone()
	modelname = result[0]
	beginind = int(result[1])
	#beginind = 0
	model_def = 'net.prototxt'
	#model_def = '/home/dilu/triple_rgbd/gyd/net_d.prototxt'
	#model_weight = '/data/rgbd_data/getRGBD/face/initial_rbm/land_t_iter_32000.caffemodel'
	model_weight = str('../snaps/'+modelname)
	print model_weight
	
	f = open(model_weight,'rb')
	caffemodel = caffe_pb2.NetParameter()
	caffemodel.ParseFromString(f.read())
	f.close()
	outpara = np.zeros([128,1024],'float32')
	outbias = np.zeros([128],'float32')
	
	for alayer in caffemodel.layer:
		if alayer.name == 'output':
			#print alayer.blobs[0].data[0:]
			#print alayer.blobs[0].data.shape
			for i in range(128):
				#print alayer.blobs[0].data[i*1024:(i+1)*1024]
				for j in range(1024):
					outpara[i][j] = alayer.blobs[0].data[i*1024+j]
				outbias[i] = alayer.blobs[1].data[i]	
	net = caffe.Net(model_def,model_weight,caffe.TEST)
	net.params['newoutput'][0].data.flat = outpara
	#print net.params['newoutput'][0].data[0:]
	net.params['newoutput'][1].data[...] = outbias
	#net = caffe.Net(model_def,model_weight,caffe.TEST)
	#choose 200 people randomly and extract their features
	for x in range (1):
		peolist = np.random.permutation(tcount)[:100]
		groupfeat  = {}
		for i in peolist:
			for j in range(numper[i]['num']):
				filename = namelist[numper[i]['begin']+j]
				data = np.array(Image.open(data_root+filename+'.jpg'))
				net.blobs['data'].data[...] = data-np.mean(data)
				output = net.forward()
				output_features = output['newoutput'][0,0:128].flat
				#print np.linalg.norm(output_features)*np.linalg.norm(output_features)
				output_features = output_features/np.linalg.norm(output_features)
				groupfeat[filename] = output_features

		#create triple
		newtriple = []
		insertind = 0
		for i in peolist:
			for j in range(numper[i]['num']):
				print "people:"+str(i)
				for l in range(numper[i]['num']):
					if l == j:
						continue
					anchor = namelist[numper[i]['begin']+j]
					positive = namelist[numper[i]['begin']+l]
					anchorfeat = groupfeat[anchor]
					apdist = np.linalg.norm(anchorfeat-groupfeat[positive])
					if apdist > l1-alpha:
						mindist = 1000	
						names = np.array(groupfeat.keys())				
						randind = np.random.permutation(len(names))[:100]
						randlist = names[randind]
						for x in randlist:
							if x[0:18] == anchor[0:18]:
								continue
							newdist = np.linalg.norm(anchorfeat-groupfeat[x])
							#print newdist
							if newdist < mindist and newdist > apdist-beta:
								mindist = newdist
								negative = x
						#if mindist > apdist or (apdist > l1 and mindist < 1000):
						if mindist < 1000:
						#if mindist< apdist + alpha:
							newtriple.append((anchor,positive,negative))
							cu.execute('INSERT OR REPLACE INTO ir_triple VALUES (?,?,?,?)',(beginind+insertind,anchor,positive,negative))
							insertind = insertind+1
							print "A-N: "+str(mindist)+" A-P: "+str(apdist) 
					'''
					negative = ''
					apdist = np.linalg.norm(anchorfeat-groupfeat[positive])
					count = 0
					while negative == '':
						if count > 2000:
							break
						randx= np.random.random_integers(99)
						while peolist[randx] == i:
							randx= np.random.random_integers(99)
						for z in range(numper[peolist[randx]]['num']):
						#print z
							count = count+1
							negname = namelist[numper[peolist[randx]]['begin']+z]
							andist = np.linalg.norm(anchorfeat-groupfeat[negname])
							if andist>(apdist-beta) and andist<(apdist+alpha):
								negative = negname
								break
					if negative!='':
						cu.execute('INSERT OR REPLACE INTO ir_triple VALUES (?,?,?,?)',(beginind+insertind,anchor,positive,negative))
						insertind = insertind+1
						#andist = np.linalg.norm(anchorfeat-groupfeat[negative])
					'''
				conn.commit()
				


	#for i in range(len(newtriple)):
	#	cu.execute('INSERT INTO land_triple VALUES (?,?,?,?)',(beginind+i,)+newtriple)
	cu.execute('UPDATE ir_model SET beginind = ? WHERE id = 0',(beginind+insertind,))
	conn.commit()
	print insertind
	itercount = itercount+1
	time.sleep(1200)
conn.close()
			

			



