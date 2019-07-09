
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import gc
import os
cwd = os.getcwd()
sys.path.append(cwd+'/../networks/')
import torch
import time
import datetime

import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
from data import build_train_dataset, build_test_dataset
import util
import numpy as np
import argparse
import shutil

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from alexnet import alexnet
from resnet import resnet
from tensorboardX import SummaryWriter
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',
	help='set if only CPU is available')
parser.add_argument('--data', action='store', default='/lmdb_imagenet',
	help='dataset path')
parser.add_argument('--arch', action='store', default='AlexNet',
	help='the architecture for the network: nin')
parser.add_argument('--lr', action='store', default='0.001', type=float,
	help='the intial learning rate')
parser.add_argument('--epochs', action='store', default='0',
	help='fisrt train epochs',type=int)
parser.add_argument('--retrain_epochs', action='store', default='80',
	help='re-train epochs',type=int)
parser.add_argument('--print_freq', action='store', default='10',
	help='re-train epochs',type=int)
parser.add_argument('--save_name', action='store', default='first_model',
	help='save the first trained model',type=str)
parser.add_argument('--load_name', action='store', default='first_model',
	help='load pretrained model',type=str)
parser.add_argument('--root_dir', action='store', default='../bagging_inde_alexnet/',
	help='root dir for different experiments',type=str)
parser.add_argument('--record_path', action='store', default='bagging_inde_alexnet',
	help='root dir for different experiments',type=str)
parser.add_argument('--pretrained', action='store', default=None,
	help='the path to the pretrained model')
parser.add_argument('--evaluate', action='store_true',
	help='evaluate the model')
parser.add_argument('--resume', type=str, default='../networks/alexnet.baseline.pth.tar',
	help='resume the model from checkpoint')
parser.add_argument('--partpal', 
                    default=1, type=int, help='parallel mode')
args = parser.parse_args()
print('==> Options:',args)

train_dataset_size = 1281167
test_dataset_size = 50000
class_size = 1000

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)
writer = SummaryWriter('../runs/'+ args.record_path)

class auto_incremental_recorder(object):
	def __init__(self, writer, name):
		self.writer = writer
		self.name = name
		self.step = 1
	def add(self, value):
		self.writer.add_scalar(self.name, value, self.step)
		self.step += 1

train_top1_write = auto_incremental_recorder(writer, 'train_batch/top1')
train_top5_write = auto_incremental_recorder(writer, 'train_batch/top5')
train_loss_write = auto_incremental_recorder(writer, 'train_batch/loss')
train_avg_top1_write = auto_incremental_recorder(writer, 'train/avg_top1')

test_loss_write = auto_incremental_recorder(writer, 'test/loss')
test_top1_write = auto_incremental_recorder(writer, 'test/avg_top1')

############check################
def ifornot_dir(directory):
##please give current dir
	import os
	if not os.path.exists(directory):
		os.makedirs(directory)

ifornot_dir(args.root_dir)

'''
# prepare the data
if not os.path.isfile(args.data+'/train_data'):
	 # check the data path
	raise Exception\
	 ('Please assign the correct data path with --data <DATA_PATH>')
'''

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    lr = args.lr * (0.1 ** (epoch // 25))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def save_state(state, is_best, save_name, root_dir=args.root_dir):
	print('==> Saving model ...')
	torch.save(state, root_dir+save_name+'.pth.tar')
	if is_best:
		shutil.copyfile(filename, root_dir+save_name+'_best.pth.tar')


def train(epoch, sample_weights=torch.Tensor(np.ones((50000,1))/50000.0)):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	adjust_learning_rate(optimizer, epoch)

	trainloader = build_train_dataset('sample_dataset', sample_weights)
	
	#print('==> Starting one epoch ...')
	for batch_idx, (data, target, _) in enumerate(trainloader):
		# measure data loading time
		#if batch_idx==2:
		#	break
		data_time.update(time.time() - end)

		bin_op.binarization()
		data, target = Variable(data.cuda()), Variable(target.cuda())
		optimizer.zero_grad()
		output = model(data) 
		# backwarding
		loss = criterion(output, target)
		prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
		losses.update(loss.data[0], data.size(0))
		top1.update(prec1[0], data.size(0))
		top5.update(prec5[0], data.size(0))
		#loss = (criterion_seperated(output, target)*Variable(sample_weights.cuda().float()*50000)).mean()
		loss.backward()
		# restore weights
		bin_op.restore()
		bin_op.updateBinaryGradWeight()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# write tensorboard
		train_top5_write.add(prec5[0])
		train_top1_write.add(prec1[0])
		train_loss_write.add(loss.data[0])
		train_avg_top1_write.add(top5.avg)

		if batch_idx % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
			'LR: {lr:.8f}\t'
			'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
			'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
			'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
			epoch, batch_idx, len(trainloader), lr=optimizer.param_groups[0]['lr'],batch_time=batch_time,
			data_time=data_time, loss=losses, top1=top1, top5=top5))
	gc.collect()
	return top1.avg

def eval_test(save_name, best_prec1):
	#global best_acc
	model.eval()
	test_loss = 0
	correct = 0
	bin_op.binarization()

	testloader = build_test_dataset()
	for i, (data, target) in tqdm(enumerate(testloader)):
		#if i==2:
		#	break
		data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)                         
		output = model(data)
		test_loss += criterion(output, target).data[0]
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	bin_op.restore()
	acc = 100. * correct / len(testloader.dataset)
	test_loss = (test_loss / len(testloader.dataset)) * target.size(0)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
	   test_loss, correct, len(testloader.dataset),
	   acc))

	test_loss_write.add(test_loss)

	if acc > best_prec1:
		save_state({
	        'arch': args.arch,
	        'state_dict': model.state_dict(),
	        'best_prec1': best_prec1,
	        'optimizer' : optimizer.state_dict(),
	    }, False, save_name)

	best_prec1 = max(acc, best_prec1)

	test_top1_write.add(acc)

	return best_prec1


def reset_learning_rate(optimizer, lr=0.001):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return



# prepare the options


# define the model
def reset_model():
	for m in model.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				c = float(m.weight.data[0].nelement())
				m.weight.data.normal_(0, 1./c)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					m.weight.data = m.weight.data.zero_().add(1.0)


def model_components():
	print('==> building model',args.arch,'...')
	model = alexnet('AlexNet')
	model.features = torch.nn.DataParallel(model.features)
	model.cuda()
	#load model
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)


			model.load_state_dict(checkpoint['state_dict'])

			#optimizer.load_state_dict(checkpoint['optimizer'])
			#print("=> loaded checkpoint '{}' (epoch {})"
			#	.format(args.resume, checkpoint['epoch']))
			del checkpoint
		else:
			raise Exception(args.resume+' is found.')
	else:
		print('==> Initializing model parameters ...')
		model = reset_model(model)
	cudnn.benchmark = True
	# initialize the model
	# print(model)

	optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=1e-5)
	#optimizer = optim.SGD(ps, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

	criterion = nn.CrossEntropyLoss().cuda()
	criterion_seperated = nn.CrossEntropyLoss(reduce=False).cuda()
	# define the binarization operator
	bin_op = util.BinOp(model, 'FL_Full')
	return model, optimizer, criterion, criterion_seperated, bin_op



def get_error_output(data, target, batch_sample_weights):

	data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
	output = model(data)
	#loss = (criterion_seperated(output, target)*Variable(batch_sample_weights.cuda().float())).mean()

	return output

def sample_models(boosting_iters, sample_weights, retrain_epoch=100):
	print(str(datetime.datetime.utcnow())+" Start bagging iter: "+str(boosting_iters) )
	print('===> Start retraining ...')

	best_acc = -1
	reset_learning_rate(optimizer, lr=args.lr)
	if boosting_iters == 0:
		best_acc = eval_test(str(boosting_iters), best_acc)
	else:
		reset_model()
		best_acc = -1
		for epoch in range(1, retrain_epoch+1):
			train(epoch, sample_weights)
			best_acc = eval_test(str(boosting_iters), best_acc)

def use_sampled_model(sampled_model, data, target, batch_sample_weights):
	#print('==> Load pretrained model form', sampled_model, '...')
	pretrained_model = torch.load(sampled_model)
	#best_acc = pretrained_model['best_acc']
	model.load_state_dict(pretrained_model['state_dict'])

	#model.load_state_dict(torch.load(sampled_model))
	model.eval()
	bin_op.binarization()
	return get_error_output(data, target, batch_sample_weights)


def combine_softmax_output(pred_test_i, pred_test, alpha_m_mat, i):
	#print(alpha_m_mat)
	pred_test_delta = alpha_m_mat[0][i] * pred_test_i
	#pred_test_delta = torch.sum(pred_test_delta, 0) / 10000.
	#pred_test_delta = pred_test_delta.unsqueeze(1)
	#print(pred_test_delta)
	#print(pred_test)
	pred_test = torch.add(pred_test, pred_test_delta.cpu())
	#print(pred_test)
	return pred_test

def most_common_element(pred_mat, test_batch_size):
	pred_most = []
	pred_mat = pred_mat.astype(int)
	for i in range(test_batch_size):

		counts = np.bincount(pred_mat[i,:])
		pred_most = np.append(pred_most, np.argmax(counts))

	return pred_most

def most_common_element_train(pred_mat, train_batch_size):
	pred_most = []
	pred_mat = pred_mat.astype(int)
	for i in range(train_batch_size):

		counts = np.bincount(pred_mat[i,:])
		pred_most = np.append(pred_most, np.argmax(counts))

	return pred_most


if __name__ == '__main__':

	model, optimizer, criterion, criterion_seperated, bin_op = model_components()

        already_boosted = -1
	boosting_iters = 10
	#sample_weights_new = torch.Tensor(np.ones((50000,1)) / 50000.)

		# Index of weak classifiers
	index_weak_cls = 0;

	train_batch_size = 1000

	test_batch_size = 1000

	testloader = build_test_dataset(batch_size=test_batch_size)
	trainloader = build_train_dataset('normal', None, batch_size=train_batch_size)

	total_classes = 1000

	train_dataset_size = 1281167
	test_dataset_size = 50000
        ens_start = 3
        ens_end = boosting_iters + 1

	for i in range(boosting_iters):
		print("Bagging "+str(i))
                if i > already_boosted:
		    sample_weights_new = np.random.choice(train_dataset_size, size=train_dataset_size)
		    sample_models(boosting_iters=i, sample_weights=sample_weights_new, retrain_epoch=args.retrain_epochs)
		print('%s %d-th Sample done !' % (str(datetime.datetime.utcnow()), i))
		
		index_weak_cls = index_weak_cls + 1
		#alpha_m_mat = torch.cat((alpha_m_mat, alpha_m), 1)

	print("Bagging finished!")
	pred_store = torch.Tensor();
	#sample_weights_new = torch.Tensor(np.ones((10000,1)));
	final_loss_total = []
	final_loss_total_2 = []
	
	#index_weak_cls = 5
	#use the sampled model
	for num_boost in range(ens_start, ens_end):

		#if num_boost==3:
		#	break

		final_loss_total = []
		final_loss_total_2 = []
		
		for batch_idx, (data, target) in enumerate(testloader):


			#if batch_idx==2:
			#	break


			pred_store = torch.Tensor();
			pred_test = torch.Tensor(np.zeros((test_batch_size,total_classes)))
			for i in range(num_boost):
				pred_test_i = use_sampled_model(args.root_dir+ str(i) +'.pth.tar', data, target, torch.Tensor(np.ones((test_batch_size,1))/1.0))
					#print(pred_test_i)
				pred = pred_test_i.max(1, keepdim=True)[1]
				#print(pred_store)
				#print(type(pred))
				#print(pred.cpu().float())
				pred_store = torch.cat((pred_store, pred.data.cpu().float()), 1)

				pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, np.ones((1,boosting_iters)), i)

			#pred = pred_test.max(1, keepdim=True)[1]
			#pred = pred.squeeze(1)
			pred_most = most_common_element(pred_store.numpy(), test_batch_size)
			pred_most = torch.Tensor(pred_most)
			pred_test = pred_test.max(1, keepdim=True)[1]
			pred_test = pred_test.squeeze(1)


			# Compute final loss in numpy
			#print('--------------------')
			#print(pred_store)
			np.save(args.root_dir+'pred_store',pred_store)
			#print(pred_most)
			np.save(args.root_dir+'pred_most',pred_most)
			#print(pred_test)
			np.save(args.root_dir+'pred_test',pred_test)
			#print(target)
			np.save(args.root_dir+'target',target)
			loss_final = [int(x) for x in (pred_most.numpy() != target.numpy())]
			final_loss = float(sum(loss_final)) / float(len(loss_final))
			final_loss_total = np.append(final_loss_total, final_loss)

			loss_final_2 = [int(x) for x in (pred_test.numpy() != target.numpy())]
			final_loss_2 = float(sum(loss_final_2)) / float(len(loss_final_2))
			final_loss_total_2 = np.append(final_loss_total_2, final_loss_2)

		print('--------------------'+'num_bagging: '+str(num_boost)+'--------------------')
		final_loss_print = np.mean(final_loss_total)
		print('\n Test accuracy from selected model: {:.4f} \n'.format(1-final_loss_print))

		final_loss_print_2 = np.mean(final_loss_total_2)
		print('\n Test accuracy from selected model 2: {:.4f} \n'.format(1-final_loss_print_2))

	for num_boost in range(ens_start, ens_end):

		#if num_boost==3:
		#	break


		final_loss_total = []
		final_loss_total_2 = []
		
		for batch_idx, (data, target) in enumerate(trainloader):


			
			#if batch_idx==2:
			#	break


			if len(target) < train_batch_size:
				continue
			pred_store = torch.Tensor();
			pred_test = torch.Tensor(np.zeros((train_batch_size,total_classes)))
			for i in range(num_boost):
				pred_test_i = use_sampled_model(args.root_dir+ str(i) +'.pth.tar', data, target, torch.Tensor(np.ones((train_batch_size,1))/1.0))
					#print(pred_test_i)
				pred = pred_test_i.max(1, keepdim=True)[1]
				#print(pred_store)
				#print(type(pred))
				#print(pred.cpu().float())
				pred_store = torch.cat((pred_store, pred.data.cpu().float()), 1)

				pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, np.ones((1,boosting_iters)), i)

			#pred = pred_test.max(1, keepdim=True)[1]
			#pred = pred.squeeze(1)
			pred_most = most_common_element_train(pred_store.numpy(), train_batch_size)
			pred_most = torch.Tensor(pred_most)
			pred_test = pred_test.max(1, keepdim=True)[1]
			pred_test = pred_test.squeeze(1)


			# Compute final loss in numpy
			#print('--------------------')
			#print(pred_store)
			np.save(args.root_dir+'pred_store',pred_store)
			#print(pred_most)
			np.save(args.root_dir+'pred_most',pred_most)
			#print(pred_test)
			np.save(args.root_dir+'pred_test',pred_test)
			#print(target)
			np.save(args.root_dir+'target',target)
			loss_final = [int(x) for x in (pred_most.numpy() != target.numpy())]
			final_loss = float(sum(loss_final)) / float(len(loss_final))
			final_loss_total = np.append(final_loss_total, final_loss)

			loss_final_2 = [int(x) for x in (pred_test.numpy() != target.numpy())]
			final_loss_2 = float(sum(loss_final_2)) / float(len(loss_final_2))
			final_loss_total_2 = np.append(final_loss_total_2, final_loss_2)

		print('--------------------'+'num_bagging: '+str(num_boost)+'--------------------')
		final_loss_print = np.mean(final_loss_total)
		print('\n Train accuracy from selected model: {:.4f} \n'.format(1-final_loss_print))

		final_loss_print_2 = np.mean(final_loss_total_2)
		print('\n Train accuracy from selected model 2: {:.4f} \n'.format(1-final_loss_print_2))
	os._exit(0)

