
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
parser.add_argument('--lr', action='store', default='0.001',
	help='the intial learning rate', type=float)
parser.add_argument('--epochs', action='store', default='0',
	help='fisrt train epochs',type=int)
parser.add_argument('--retrain_epochs', action='store', default='80',
	help='re-train epochs',type=int)
parser.add_argument('--print_freq', action='store', default='1',
	help='re-train epochs',type=int)
parser.add_argument('--save_name', action='store', default='first_model',
	help='save the first trained model',type=str)
parser.add_argument('--load_name', action='store', default='first_model',
	help='load pretrained model',type=str)
parser.add_argument('--root_dir', action='store', default='../adam_log_1e-3_alex_80_reset_SAMME_new_512b/',
	help='root dir for different experiments',type=str)
parser.add_argument('--record_path', action='store', default='adam_log_1e-3_alex_80_reset_SAMME_new_512b/',
	help='root dir for different experiments',type=str)
parser.add_argument('--boosting_mode', action='store', default='logit',
	help='boosting mode to use', type=str)
parser.add_argument('--pretrained', action='store', default=None,
	help='the path to the pretrained model')
parser.add_argument('--evaluate', action='store_true',
	help='evaluate the model')
parser.add_argument('--resume', type=str, default='../networks/alexnet.baseline.pth.tar',
	help='resume the model from checkpoint')
parser.add_argument('--partpal', 
                    default=0, type=int, help='parallel mode')
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

test_loss_write = auto_incremental_recorder(writer, 'test/loss')
test_top1_write = auto_incremental_recorder(writer, 'train/avg_top1')
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

'''
def adjust_learning_rate(optimizer, epoch):
	lr = float(args.lr) * (0.1 ** int(epoch / 2))
	print('Learning rate:', lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
'''


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


def train(epoch, scheduler, sample_weights=torch.Tensor(np.ones((50000,1))/50000.0)):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	#adjust_learning_rate(optimizer, epoch)

	trainloader = build_train_dataset('sample_batch', sample_weights)
	
	#print('==> Starting one epoch ...')
	for batch_idx, (data, target, _) in enumerate(trainloader):
		# measure data loading time
		#if batch_idx == 2500:
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
	scheduler.step()
	return top1.avg
'''
def test(save_name, best_prec1, testloader_in):

	#global best_acc

	model.eval()
	test_loss = 0
	correct = 0
	bin_op.binarization()

	for i, (data, target, _) in tqdm(enumerate(testloader_in)):
		if i ==1:
			break
		data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)                         
		output = model(data)
		test_loss += criterion(output, target).data[0]
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	bin_op.restore()
	acc = 100. * correct / len(testloader_in.dataset)
	test_loss /= len(testloader_in.dataset)
	print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
	   test_loss * 1000, correct, len(testloader_in.dataset),
	   100. * correct / len(testloader_in.dataset)))

	#is_best = prec1 > best_prec1
	
	if acc > best_prec1:
		save_state({
	        'arch': args.arch,
	        'state_dict': model.state_dict(),
	        'best_prec1': best_prec1,
	        'optimizer' : optimizer.state_dict(),
	    }, False, save_name)

	best_prec1 = max(acc, best_prec1)
	return best_prec1
'''
def eval_test(save_name, best_prec1, now_prec1):
	#global best_acc
	model.eval()
	test_loss = 0
	correct = 0
	bin_op.binarization()

	testloader = build_test_dataset()
	for i, (data, target) in tqdm(enumerate(testloader)):
		#if i==3:
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
	

	acc = now_prec1
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

	model = alexnet(args.arch)

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

	# define solver and criterio
	ps = filter(lambda x: x.requires_grad, model.parameters())

	optimizer = optim.Adam(ps, lr=args.lr, weight_decay=0.00001)
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




def update_weights_mild(softmax_output, target, sample_weights):
	'''
	data: in pytorch tensor
	target: in pytorch tensor
	sample_weights = in numpy which has the same size as data in first dimension

	'''
	# update the weights according to Adaboost
	print("start updating..")
	pred_numpy = softmax_output
	target_numpy = target
	pred_numpy = torch.squeeze(pred_numpy)
	# Tensor version    
	miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
	miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
	miss = miss.unsqueeze(1)

	#print('miss',miss)
	#print('miss2',miss2)
	err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
	alpha_m = (torch.log((1 - err_m) / float(err_m)) + torch.log(torch.Tensor([999])))

	#print('err_m',err_m)
	#print('alpha_m',alpha_m)
	prior_exp = torch.t(torch.Tensor(alpha_m * miss2))
	sample_weights_new = (1 - sample_weights) * torch.exp(-prior_exp)
	sample_weights_new = (train_dataset_size/class_size) * sample_weights_new / torch.sum(torch.abs(sample_weights_new))
	sample_weights_new = 1 - sample_weights_new

	print(torch.min(sample_weights_new))
	print(torch.max(sample_weights_new))
	print(sample_weights_new)
	print("weights updated!")
	return sample_weights_new, alpha_m

def update_weights_logit(softmax_output, target, sample_weights):
	'''
	data: in pytorch tensor
	target: in pytorch tensor
	sample_weights = in numpy which has the same size as data in first dimension

	'''
	# update the weights according to Adaboost
	print("start updating..")
	pred_numpy = softmax_output
	target_numpy = target
	pred_numpy = torch.squeeze(pred_numpy)
	# Tensor version    
	miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
	miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
	miss = miss.unsqueeze(1)

	err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
	alpha_m = 0.1*(torch.log((1 - err_m) / float(err_m)) + torch.log(torch.Tensor([999])))

	prior_exp = torch.t(torch.Tensor(alpha_m * miss2))
	sample_weights_new = 1 / ((1 / sample_weights - 1) * torch.exp(-prior_exp) + 1)
	print(torch.min(sample_weights_new))
	print(torch.max(sample_weights_new))
	#print(sample_weights_new)
	print("weights updated!")
	return sample_weights_new, alpha_m


def update_weights_SAMME(softmax_output, target, sample_weights):
	'''
	data: in pytorch tensor
	target: in pytorch tensor
	sample_weights = in numpy which has the same size as data in first dimension

	'''
	# update the weights according to Adaboost
	print("start updating..")
	pred_numpy = softmax_output
	target_numpy = target
	pred_numpy = torch.squeeze(pred_numpy)
	# Tensor version    
	miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
	miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
	miss = miss.unsqueeze(1)

	err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
	alpha_m = 0.001 * (torch.log((1 - err_m) / float(err_m)) + torch.log(torch.Tensor([999])))

	prior_exp = torch.t(torch.Tensor(alpha_m * miss2))
	sample_weights_new = sample_weights * torch.exp(prior_exp)
	print(torch.min(sample_weights_new))
	print(torch.max(sample_weights_new))
	print(sample_weights_new)
	print("weights updated!")
	return sample_weights_new, alpha_m



def sample_models(boosting_iters, sample_weights, retrain_epoch=100):
	print(str(datetime.datetime.utcnow())+" Start boosting iter: "+str(boosting_iters) )
	print('===> Start retraining ...')

	best_acc = -1
	reset_learning_rate(optimizer, lr=args.lr)
	#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, retrain_epoch, eta_min=0, last_epoch=-1)
	#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [3], gamma=0.1)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1) # this is jiecao's implement
	if boosting_iters == 0:
		#train_loader_out = build_train_dataset('sample_batch', sample_weights)
		#best_acc = test(str(boosting_iters), best_acc, train_loader_out)
		#eval_test()
		best_acc = eval_test(str(boosting_iters), best_acc, 0)
	else:
		reset_model()
		for epoch in range(1, retrain_epoch+1):
			now_acc = train(epoch, scheduler, sample_weights)
			#if (epoch % 2 ==0) & (epoch > 0.7*retrain_epoch):
			print("Epoch " + str(epoch) + ": " + str(now_acc))
			if epoch % 1 ==0:
				#best_acc = test(str(boosting_iters), best_acc, train_loader_out)
				#eval_test()
				best_acc = eval_test(str(boosting_iters), best_acc, now_acc)
		#print('Retraining Done ...')

	pretrained_model = torch.load(args.root_dir+ str(boosting_iters) +'.pth.tar')
	#best_acc = pretrained_model['best_acc']
	model.load_state_dict(pretrained_model['state_dict'])

	model.eval()
	bin_op.binarization()

	
	pred_output = torch.Tensor(np.zeros((train_dataset_size, 1))) #torch tensor in cpu
	label_in_tensor = torch.Tensor(np.zeros((train_dataset_size, ))) #torch tensor in cpu

	trainloader = build_train_dataset('normal', None, batch_size=1000)
	for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
	#print(str(batch_idx)+'-th test batch')
	#	if batch_idx ==3:
	#		break
		batch_size = target.size(0)
		batch_sample_weights = sample_weights[batch_idx*batch_size:(batch_idx+1)*batch_size]

		batch_softmax_output = get_error_output(data, target, batch_sample_weights)
		pred_output[batch_idx*batch_size:(batch_idx+1)*batch_size,:] = batch_softmax_output.max(1, keepdim=True)[1].data.cpu()
		label_in_tensor[batch_idx*batch_size:(batch_idx+1)*batch_size] = target
	
	bin_op.restore()	
	np.save(args.root_dir+'pred_output_'+str(boosting_iters), pred_output.numpy())
	np.save(args.root_dir+'label_in_tensor_'+str(boosting_iters), label_in_tensor.numpy())
	'''
	pred_output = torch.Tensor(np.load(args.root_dir+'pred_output_'+str(boosting_iters)+'.npy'))
	label_in_tensor = torch.Tensor(np.load(args.root_dir+'label_in_tensor_'+str(boosting_iters)+'.npy'))
	'''

	if args.boosting_mode == 'mild':
		best_sample_weights, best_alpha_m = update_weights_mild(pred_output, label_in_tensor, sample_weights)
	elif args.boosting_mode == 'logit':
		best_sample_weights, best_alpha_m = update_weights_logit(pred_output, label_in_tensor, sample_weights)
	elif args.boosting_mode == 'SAMME':
		best_sample_weights, best_alpha_m = update_weights_SAMME(pred_output, label_in_tensor, sample_weights)
	else:
		raise ValueError('Wrong Boosting Mode !')
	np.save(args.root_dir+'sample_weights_'+str(boosting_iters), best_sample_weights.numpy())
	return best_sample_weights, best_alpha_m

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

def most_common_element(pred_mat, alpha_m_mat, num_boost, test_batch_size, total_classes):
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(test_batch_size):
        best_value = -1000
        best_pred = -1
        #counts = np.bincount(pred_mat[i,:])
        #pred_most = np.append(pred_most, np.argmax(counts))
        for j in range(total_classes):
            #print(pred_mat[i,:])
            #print(j*np.ones((num_boost,), dtype=int))
            mask = [int(x) for x in (pred_mat[i,:] == j*np.ones((num_boost,), dtype=int))]
            #print(mask)
            #print(alpha_m_mat[0][5:5+num_boost])
            if np.sum(mask * alpha_m_mat[0][0:0+num_boost]) > best_value:
                best_value = np.sum(mask * alpha_m_mat[0][0:0+num_boost])
                best_pred = j
        pred_most = np.append(pred_most, best_pred)

    return pred_most

def most_common_element_train(pred_mat, alpha_m_mat, num_boost, train_batch_size, total_classes):
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(train_batch_size):
        best_value = -1000
        best_pred = -1
        #counts = np.bincount(pred_mat[i,:])
        #pred_most = np.append(pred_most, np.argmax(counts))
        for j in range(total_classes):
            #print(pred_mat[i,:])
            #print(j*np.ones((num_boost,), dtype=int))
            mask = [int(x) for x in (pred_mat[i,:] == j*np.ones((num_boost,), dtype=int))]
            #print(mask)
            #print(alpha_m_mat[0][5:5+num_boost])
            if np.sum(mask * alpha_m_mat[0][0:0+num_boost]) > best_value:
                best_value = np.sum(mask * alpha_m_mat[0][0:0+num_boost])
                best_pred = j
        pred_most = np.append(pred_most, best_pred)

    return pred_most


if __name__ == '__main__':
		#test
	model, optimizer, criterion, criterion_seperated, bin_op = model_components()

	train_batch_size = 1000

	test_batch_size = 1000

	total_classes = 1000


	
	best_acc = 0
	if args.epochs!=0:
		#print(type(args.epochs))
		for epoch in range(1, int(args.epochs)+1):
			train(epoch)
			best_acc = test(args.save_name, best_acc)
	

	boosting_iters = 5
	already_boosted = 0
	sample_weights_new = torch.Tensor(np.ones((train_dataset_size,1))/(train_dataset_size+0.1))

		# Index of weak classifiers
	index_weak_cls = 0;

	ens_start = 3
	ens_end = boosting_iters
		# Store alpha_m
	alpha_m_mat = torch.Tensor();

	# Update sample_weights
	for i in range(boosting_iters):
		print("boosting"+str(i))
		
		sample_weights_new, alpha_m = sample_models(boosting_iters=i, sample_weights=sample_weights_new, retrain_epoch=args.retrain_epochs)
		print('%s %d-th Sample done !' % (str(datetime.datetime.utcnow()), i))
		#alpha_m = torch.Tensor(1).resize_(1,1)
		
		index_weak_cls = index_weak_cls + 1
		alpha_m_mat = torch.cat((alpha_m_mat, alpha_m), 1)
		np.save(args.root_dir+'alpha_m_mat',alpha_m_mat.numpy())
        #alpha_m_mat = torch.Tensor(np.load(args.root_dir+'alpha_m_mat.npy'))
	print(alpha_m_mat)
        print("boosting finished!")
	print(torch.max(sample_weights_new))
	print(torch.min(sample_weights_new))
	pred_store = torch.Tensor();
	#sample_weights_new = torch.Tensor(np.ones((10000,1)));

	testloader = build_test_dataset(batch_size = test_batch_size)
	trainloader = build_train_dataset('normal', None, batch_size = train_batch_size)
	
	#index_weak_cls = 5
	#use the sampled model
	for num_boost in range(ens_start, ens_end):

		#if num_boost == 3:
		#	break


		final_loss_total = []
		final_loss_total_2 = []
		for i, (data, target) in enumerate(testloader):
			#if i == 2 :
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

				pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, alpha_m_mat, i)

			#pred = pred_test.max(1, keepdim=True)[1]
			#pred = pred.squeeze(1)
			pred_most = most_common_element(pred_store.numpy(), alpha_m_mat.numpy(), num_boost, test_batch_size, total_classes)
			pred_most = torch.Tensor(pred_most)
			pred_test = pred_test.max(1, keepdim=True)[1]
			pred_test = pred_test.squeeze(1)


			# Compute final loss in numpy
			
			#print(pred_store)
			np.save('pred_store',pred_store)
			#print(pred_most)
			np.save('pred_most',pred_most)
			#print(pred_test)
			np.save('pred_test',pred_test)
			#print(target)
			np.save('target',target)
			loss_final = [int(x) for x in (pred_most.numpy() != target.numpy())]
			final_loss = float(sum(loss_final)) / float(len(loss_final))
			final_loss_total = np.append(final_loss_total, final_loss)

			loss_final_2 = [int(x) for x in (pred_test.numpy() != target.numpy())]
			final_loss_2 = float(sum(loss_final_2)) / float(len(loss_final_2))
			final_loss_total_2 = np.append(final_loss_total_2, final_loss_2)

		print('--------------------'+'num_boost: '+str(num_boost)+'--------------------')
		final_loss_print = np.mean(final_loss_total)
		print('\n Test accuracy from selected model: {:.4f} \n'.format(1-final_loss_print))

		final_loss_print_2 = np.mean(final_loss_total_2)
		print('\n Test accuracy from selected model 2: {:.4f} \n'.format(1-final_loss_print_2))

	#Train loader
	for num_boost in range(ens_start, ens_end):

		#if num_boost==3:
		#	break


		final_loss_total = []
		final_loss_total_2 = []
		for i, (data, target) in enumerate(trainloader):

			#if i==2:
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

				pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, alpha_m_mat, i)

			#pred = pred_test.max(1, keepdim=True)[1]
			#pred = pred.squeeze(1)
			pred_most = most_common_element_train(pred_store.numpy(), alpha_m_mat.numpy(), num_boost, train_batch_size, total_classes)
			pred_most = torch.Tensor(pred_most)
			pred_test = pred_test.max(1, keepdim=True)[1]
			pred_test = pred_test.squeeze(1)


			# Compute final loss in numpy
			
			#print(pred_store)
			np.save('pred_store',pred_store)
			#print(pred_most)
			np.save('pred_most',pred_most)
			#print(pred_test)
			np.save('pred_test',pred_test)
			#print(target)
			np.save('target',target)
			loss_final = [int(x) for x in (pred_most.numpy() != target.numpy())]
			final_loss = float(sum(loss_final)) / float(len(loss_final))
			final_loss_total = np.append(final_loss_total, final_loss)

			loss_final_2 = [int(x) for x in (pred_test.numpy() != target.numpy())]
			final_loss_2 = float(sum(loss_final_2)) / float(len(loss_final_2))
			final_loss_total_2 = np.append(final_loss_total_2, final_loss_2)

		print('--------------------'+'num_boost: '+str(num_boost)+'--------------------')
		final_loss_print = np.mean(final_loss_total)
		print('\n Train accuracy from selected model: {:.4f} \n'.format(1-final_loss_print))

		final_loss_print_2 = np.mean(final_loss_total_2)
		print('\n Train accuracy from selected model 2: {:.4f} \n'.format(1-final_loss_print_2))
	os._exit(0)

