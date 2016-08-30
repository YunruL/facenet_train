import os, sys
import argparse
import time
import numpy as np
caffe_root = '/home/liuyunr/hha_pair/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import google.protobuf as pb2
from PIL import Image
import sqlite3

class ModelTrainer:
	def __init__(self, solver_prototxt, mean_ = None, pretrained_model = None, gpu_id = 4, data_root = './data', cu = None):
		if gpu_id >= 0:
			caffe.set_device(gpu_id)
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()
		self.model_dir = os.path.dirname(solver_prototxt)
		self.solver = caffe.SGDSolver(solver_prototxt)
		if pretrained_model is not None:
			self.solver.net.copy_from(pretrained_model)
		self.solver_param = caffe.proto.caffe_pb2.SolverParameter()
		with open(solver_prototxt, 'rt') as fd:
			pb2.text_format.Merge(fd.read(),self.solver_param)
		#self.transformer = caffe.io.Transformer({'data':self.solver.net.blobs['data'].data.shape})
		if mean_ is not None:
		#	self.transformer.set_mean('data',np.load(mean_file))  # need to transform the mean_file to .npy
			self.im_mean = mean_
		else:
			self.im_mean = None
		self.data_root = data_root
		
		self.train_batch_size = self.solver.net.blobs['data'].data.shape[0]
		self.test_batch_size = self.solver.test_nets[0].blobs['data'].data.shape[0]
		
		#self.train_triple = [x.strip().split(' ') for x in open('/home/dilu/triple_rgbd/gyd/choosetriple/triple_land06232.txt').readlines()]  #which file?
		self.test_triple = [x.strip().split(' ') for x in open('validation.txt').readlines()]
		self.cu = cu
		#self.train_len = len(self.train_triple)
	
	def read_triple(self):
		#self.cu.execute('SELECT beginind FROM land_model where id = 0')
		#result = self.cu.fetchone()
		#beginind = int(result[0])
	   try:
		cu.execute('SELECT count(*) FROM ir_triple')
		result = cu.fetchone()
		amount = int(result[0])
		beginind = amount - 50000
		self.cu.execute('SELECT anchor,positive,negative FROM ir_triple where id > ?',(beginind,))
		result = self.cu.fetchall()
		self.train_triple = [np.array(x) for x in result]
		self.train_len = len(self.train_triple)
	
	   finally:
    		pass	
	def read_images(self, img_list):
		batch_size, channels, height, width = self.solver.net.blobs['data'].data.shape
		X = np.zeros([len(img_list),channels,height,width], dtype = np.float32)
		i = 0
		for img in img_list:
			data = np.load(data_root+img[0]+'.npy')
                        data_p = np.load(data_root+img[1]+'.npy')
                        data_n = np.load(data_root+img[2]+'.npy')


			X[i,0,:,:] = data
			X[i,1,:,:] = data_p
			X[i,2,:,:] = data_n
			i = i+1
		return X
		
	# prepare random or pre-defined batch_size data
	def prepare_batch_data(self, phase = 'train', idx = None, batch_num = 1):
		shufmax = max(1000,batch_num*self.train_batch_size)
		if phase == 'train':
			timex = time.time()
			if idx is not None:
				train_idx = idx
			else:
				assert(batch_num*self.train_batch_size<self.train_len)
				begins = np.random.random_integers(self.train_len-shufmax)
				train_idx = np.random.permutation(shufmax)[:self.train_batch_size*batch_num]
				train_idx = train_idx+begins
				
			#Here is the list of data in a batch
			chosentrain = []
			xsize = len(train_idx)
			for i in train_idx:
				chosentrain.append(self.train_triple[i])
			#Here should be corresponding triplet data
			self.train_data = self.read_images(chosentrain)
			#print self.train_A
			self.train_label = np.ones(xsize,dtype = np.float32)

			#print time.time()-timex
			self.solver.net.set_input_arrays(self.train_data,self.train_label)
			#print self.solver.net.blobs['anchor'].data
			
		elif phase == 'test':
			if idx is not None:
				test_idx = idx
			else:
				test_idx = np.arange(len(self.test_triple))[:self.test_batch_size*batch_num]
			chosentest = []
			for i in test_idx:
				chosentest.append(self.test_triple[i])
			xsize = len(test_idx)
			self.data = self.read_images(chosentest)
			self.label = np.ones(xsize,dtype = np.float32)
			#print self.test_A.shape
			self.solver.test_nets[0].set_input_arrays(self.data,self.label)
			
	
	def train_model(self):
		t1 = time.time()
		#assert(self.solver_param.average_loss>=1)
		while self.solver.iter < self.solver_param.max_iter:
		   try:	
			if self.solver.iter % 1000 == 0:
				self.read_triple()
			if self.solver.iter % 2000 == 0:
				if self.solver.iter > 0:
					prefix = self.solver_param.snapshot_prefix.split('/')
					modelname = prefix[len(prefix)-1]+'_iter_'+str(self.solver.iter)+'.caffemodel'
					self.cu.execute('UPDATE ir_model SET modelname = ? WHERE id = 0',(modelname,))
		   finally:
			if self.solver.iter % 1 ==0:
				t3 = time.time() 
				#this is small batch test
				#train_idx = np.random.permutation(100)[:35]
				self.prepare_batch_data('train', batch_num = 1) #batch_num = self.solver_param.average_loss
				t4 = time.time()
				#print self.solver.net.blobs['anchor'].data
				#print 'reading batch time: %f' %(t4-t3)
			self.solver.step(1)  #n should be the batch num
			#print 'solve a step'
			if self.solver.iter % self.solver_param.display == 0:
				t2 = time.time()
				print 'speed: {:.3f}s / iter'.format((t2-t1)/self.solver_param.display)
				t1 = t2
			if self.solver.iter % (self.solver_param.test_interval) == 0:
				print '#############  Test Begin   ##############'
				t5 = time.time()
				test_num = len(self.test_triple)
				test_num_mini = 7000
				iter_num = int(np.ceil(test_num_mini*1.0/self.test_batch_size))-1
				s1 = 0.0
				randids = np.random.permutation(test_num)
				for i in xrange(iter_num):
					
					if (i+1)*self.test_batch_size>test_num:
						break
						#ids = np.hstack([randids[i*self.test_batch_size:test_num],randids[0,(i+1)*self.test_batch_size%test_num]])
					else:
						ids = randids[i*self.test_batch_size:(i+1)*self.test_batch_size]
					#print ids
					self.prepare_batch_data('test',ids)
					self.solver.test_nets[0].forward()
					s1 += self.solver.test_nets[0].blobs['loss'].data.item() #here is the analysis of the loss layer, change to the version you need
					
				s1 /= iter_num
				print 'loss: %f' % s1
				t6 = time.time()
				print '#############  Test Ends in %f seconds   ##############' % (t6-t5)
			#if self.solver.iter % 1000 == 0: #save a model

if __name__ == '__main__':
	#define the training parameters here
	#tmp = sys.stdout
	#now = str(time.time())
	#sys.stdout = open('/home/dilu/triple_rgbd/gyd/caffelog/'+now+'.txt','wt')
	conn = sqlite3.connect('oltriple.db')
	conn.isolation_level = None
	cu = conn.cursor()
	parser = argparse.ArgumentParser(description = 'Train a caffe model')
	parser.add_argument('--gpu_id', dest='gpu_id', help='GPU device to use[0]', default=6, type=int)
	args = parser.parse_args()
	
	solver_prototxt = 'ir_solver.prototxt'
	pretrained_model = '../snaps/l1l2_iter_30000.caffemodel'
	assert(os.path.exists(solver_prototxt) and os.path.exists(pretrained_model))
	
	im_mean = np.load('../ir_mean.npy')
	data_root = '/data/liuyunr/ir_mean/'
	trainer = ModelTrainer( solver_prototxt, 
				mean_ = im_mean,
				pretrained_model = pretrained_model,
				#pretrained_model = None,
				gpu_id = 6,
				data_root = data_root,
				cu = cu)
	trainer.train_model()
	conn.close()
	#sys.stdout.close()
	#sys.stdout = tmp
			
			
			
			
	
	
	
			
