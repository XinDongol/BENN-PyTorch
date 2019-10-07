
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import caffe
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
from resnet_new import resnet
from tensorboardX import SummaryWriter
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',
	help='set if only CPU is available')
parser.add_argument('--data', action='store', default='/lmdb_imagenet',
	help='dataset path')
parser.add_argument('--arch', action='store', default='resnet',
	help='the architecture for the network: nin')
parser.add_argument('--lr', action='store', default='0.001',
	help='the intial learning rate', type=float)
parser.add_argument('--epochs', action='store', default='0',
	help='fisrt train epochs',type=int)
parser.add_argument('--retrain_epochs', action='store', default='80',
	help='re-train epochs',type=int)
parser.add_argument('--print_freq', action='store', default='1000',
	help='re-train epochs',type=int)
parser.add_argument('--save_name', action='store', default='first_model',
	help='save the first trained model',type=str)
parser.add_argument('--load_name', action='store', default='first_model',
	help='load pretrained model',type=str)
parser.add_argument('--root_dir', action='store', default='../boost_inde_resnet_release/',
	help='root dir for different experiments',type=str)
parser.add_argument('--record_path', action='store', default='boost_inde_resnet_release/',
	help='root dir for different experiments',type=str)
parser.add_argument('--boosting_mode', action='store', default='C',
	help='boosting mode to use', type=str)
parser.add_argument('--pretrained', action='store', default=None,
	help='the path to the pretrained model')
parser.add_argument('--evaluate', action='store_true',
	help='evaluate the model')
parser.add_argument('--resume', type=str, default='../networks/resnet_model_best.pth.tar',
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

def eval_test(save_name, best_prec1, now_prec1):
	#global best_acc
	model.eval()
	test_loss = 0
	correct = 0
	bin_op.binarization()

	testloader = build_test_dataset()
	for i, (data, target) in tqdm(enumerate(testloader)):
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

	model = resnet('ResNet_imagenet', pretrained=args.pretrained, num_classes=1000, depth=18, dataset='imagenet')
	#load model
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			new_params = model.state_dict()
			new_params.update(checkpoint['state_dict'])
			model.load_state_dict(new_params)
			del checkpoint
		else:
			raise Exception(args.resume+' is found.')
	else:
		print('==> Initializing model parameters ...')
		for m in model.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				c = float(m.weight.data[0].nelement())
				m.weight.data.normal_(0, 1./c)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					m.weight.data = m.weight.data.zero_().add(1.0)

	#data parallel
	model = torch.nn.DataParallel(model).cuda()
	cudnn.benchmark = True

	# define solver and criterio
	ps = filter(lambda x: x.requires_grad, model.parameters())

	optimizer = optim.Adam(ps, lr=args.lr, weight_decay=0.00001)

	criterion = nn.CrossEntropyLoss().cuda()
	criterion_seperated = nn.CrossEntropyLoss(reduce=False).cuda()
	# define the binarization operator
	bin_op = util.BinOp(model, 'FL_Full')
	return model, optimizer, criterion, criterion_seperated, bin_op



def get_error_output(data, target, batch_sample_weights):

	data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
	output = model(data)

	return output




def boostA(softmax_output, target, sample_weights):
	print("start updating..")
	pred_numpy = softmax_output
	target_numpy = target
	pred_numpy = torch.squeeze(pred_numpy)
	miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
	miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
	miss = miss.unsqueeze(1)
	err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
	alpha_m = (torch.log((1 - err_m) / float(err_m)) + torch.log(torch.Tensor([999])))
	prior_exp = torch.t(torch.Tensor(alpha_m * miss2))
	sample_weights_new = (1 - sample_weights) * torch.exp(-prior_exp)
	sample_weights_new = (train_dataset_size/class_size) * sample_weights_new / torch.sum(torch.abs(sample_weights_new))
	sample_weights_new = 1 - sample_weights_new
	print("weights updated!")
	return sample_weights_new, alpha_m

def boostB(softmax_output, target, sample_weights):
	print("start updating..")
	pred_numpy = softmax_output
	target_numpy = target
	pred_numpy = torch.squeeze(pred_numpy)  
	miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
	miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
	miss = miss.unsqueeze(1)
	err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
	alpha_m = (torch.log((1 - err_m) / float(err_m)) + torch.log(torch.Tensor([999])))
	prior_exp = torch.t(torch.Tensor(alpha_m * miss2))
	sample_weights_new = 1 / ((1 / sample_weights - 1) * torch.exp(-prior_exp) + 1)
	print("weights updated!")
	return sample_weights_new, alpha_m


def boostC(softmax_output, target, sample_weights):
	print("start updating..")
	pred_numpy = softmax_output
	target_numpy = target
	pred_numpy = torch.squeeze(pred_numpy) 
	miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
	miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
	miss = miss.unsqueeze(1)
	err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
	alpha_m = 0.001 * (torch.log((1 - err_m) / float(err_m)) + torch.log(torch.Tensor([999])))
	prior_exp = torch.t(torch.Tensor(alpha_m * miss2))
	sample_weights_new = sample_weights * torch.exp(prior_exp)
	print("weights updated!")
	return sample_weights_new, alpha_m



def sample_models(boosting_iters, sample_weights, retrain_epoch=100):
	print(str(datetime.datetime.utcnow())+" Start boosting iter: "+str(boosting_iters) )
	print('===> Start retraining ...')

	best_acc = -1
	reset_learning_rate(optimizer, lr=args.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1) # this is jiecao's implement
	if boosting_iters == 0:
		best_acc = eval_test(str(boosting_iters), best_acc, 0)
	else:
		reset_model()
		for epoch in range(1, retrain_epoch+1):
			now_acc = train(epoch, scheduler, sample_weights)
			print("Epoch " + str(epoch) + ": " + str(now_acc))
			if epoch % 1 ==0:
				best_acc = eval_test(str(boosting_iters), best_acc, now_acc)

	pretrained_model = torch.load(args.root_dir+ str(boosting_iters) +'.pth.tar')
	model.load_state_dict(pretrained_model['state_dict'])

	model.eval()
	bin_op.binarization()

	
	pred_output = torch.Tensor(np.zeros((train_dataset_size, 1))) #torch tensor in cpu
	label_in_tensor = torch.Tensor(np.zeros((train_dataset_size, ))) #torch tensor in cpu

	trainloader = build_train_dataset('normal', None, batch_size=256)
	for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
		batch_size = target.size(0)
		batch_sample_weights = sample_weights[batch_idx*batch_size:(batch_idx+1)*batch_size]

		batch_softmax_output = get_error_output(data, target, batch_sample_weights)
		pred_output[batch_idx*batch_size:(batch_idx+1)*batch_size,:] = batch_softmax_output.max(1, keepdim=True)[1].data.cpu()
		label_in_tensor[batch_idx*batch_size:(batch_idx+1)*batch_size] = target
	
	bin_op.restore()	
	np.save(args.root_dir+'pred_output_'+str(boosting_iters), pred_output.numpy())
	np.save(args.root_dir+'label_in_tensor_'+str(boosting_iters), label_in_tensor.numpy())

	if args.boosting_mode == 'A':
		best_sample_weights, best_alpha_m = boostA(pred_output, label_in_tensor, sample_weights)
	elif args.boosting_mode == 'B':
		best_sample_weights, best_alpha_m = boostB(pred_output, label_in_tensor, sample_weights)
	elif args.boosting_mode == 'C':
		best_sample_weights, best_alpha_m = boostC(pred_output, label_in_tensor, sample_weights)
	else:
		raise ValueError('Wrong Boosting Mode !')
	np.save(args.root_dir+'sample_weights_'+str(boosting_iters), best_sample_weights.numpy())
	return best_sample_weights, best_alpha_m

def use_sampled_model(sampled_model, data, target, batch_sample_weights):
	pretrained_model = torch.load(sampled_model)
	model.load_state_dict(pretrained_model['state_dict'])
	model.eval()
	bin_op.binarization()
	return get_error_output(data, target, batch_sample_weights)


def combine_softmax_output(pred_test_i, pred_test, alpha_m_mat, i):
	pred_test_delta = alpha_m_mat[0][i] * pred_test_i
	pred_test = torch.add(pred_test, pred_test_delta.cpu())
	return pred_test

def most_common_element(pred_mat, alpha_m_mat, num_boost, test_batch_size, total_classes):
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(test_batch_size):
        best_value = -1000
        best_pred = -1
        for j in range(total_classes):
            mask = [int(x) for x in (pred_mat[i,:] == j*np.ones((num_boost,), dtype=int))]
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
        for j in range(total_classes):
            mask = [int(x) for x in (pred_mat[i,:] == j*np.ones((num_boost,), dtype=int))]
            if np.sum(mask * alpha_m_mat[0][0:0+num_boost]) > best_value:
                best_value = np.sum(mask * alpha_m_mat[0][0:0+num_boost])
                best_pred = j
        pred_most = np.append(pred_most, best_pred)

    return pred_most

def get_ens_test(num_boost):
	final_loss_total = []
	final_loss_total_2 = []
	for i, (data, target) in enumerate(testloader):
		pred_store = torch.Tensor()
		pred_test = torch.Tensor(np.zeros((test_batch_size,total_classes)))
		for i in range(num_boost):
			pred_test_i = use_sampled_model(args.root_dir+ str(i) +'.pth.tar', data, target, torch.Tensor(np.ones((test_batch_size,1))/1.0))
			pred = pred_test_i.max(1, keepdim=True)[1]
			pred_store = torch.cat((pred_store, pred.data.cpu().float()), 1)
			pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, alpha_m_mat, i)
		pred_most = most_common_element(pred_store.numpy(), alpha_m_mat.numpy(), num_boost, test_batch_size, total_classes)
		pred_most = torch.Tensor(pred_most)
		pred_test = pred_test.max(1, keepdim=True)[1]
		pred_test = pred_test.squeeze(1)
		# Compute final loss in numpy
		np.save('pred_store',pred_store)
		np.save('pred_most',pred_most)
		np.save('pred_test',pred_test)
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


def get_ens_train(num_boost):
	final_loss_total = []
	final_loss_total_2 = []
	for i, (data, target) in enumerate(trainloader):
		if len(target) < train_batch_size:
			continue
		pred_store = torch.Tensor()
		pred_test = torch.Tensor(np.zeros((train_batch_size,total_classes)))
		for i in range(num_boost):
			pred_test_i = use_sampled_model(args.root_dir+ str(i) +'.pth.tar', data, target, torch.Tensor(np.ones((train_batch_size,1))/1.0))
			pred = pred_test_i.max(1, keepdim=True)[1]
			pred_store = torch.cat((pred_store, pred.data.cpu().float()), 1)
			pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, alpha_m_mat, i)

		pred_most = most_common_element_train(pred_store.numpy(), alpha_m_mat.numpy(), num_boost, train_batch_size, total_classes)
		pred_most = torch.Tensor(pred_most)
		pred_test = pred_test.max(1, keepdim=True)[1]
		pred_test = pred_test.squeeze(1)
		# Compute final loss in numpy
		np.save('pred_store',pred_store)
		np.save('pred_most',pred_most)
		np.save('pred_test',pred_test)
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

if __name__ == '__main__':
	model, optimizer, criterion, criterion_seperated, bin_op = model_components()

	train_batch_size = 512

	test_batch_size = 100

	total_classes = 1000

	testloader = build_test_dataset(batch_size = test_batch_size)
	trainloader = build_train_dataset('normal', None, batch_size = train_batch_size)

	#NOTE: we should remove this because we just use pretrained model for the first boosting. 
	#---------------------------------------------
	best_acc = 0
	if args.epochs!=0:
		#print(type(args.epochs))
		for epoch in range(1, int(args.epochs)+1):
			train(epoch)
			best_acc = test(args.save_name, best_acc)
	#--------------------------------------------------
	

	boosting_iters = 3
	already_boosted = 3
	sample_weights_new = torch.Tensor(np.ones((train_dataset_size,1))/(train_dataset_size+0.001))

	# Index of weak classifiers
	index_weak_cls = 0

	# how many model you want to ensemble 
	ens_start = 3
	# Store alpha_m
	if already_boosted == 0:
		alpha_m_mat = torch.Tensor()
	else:
		alpha_m_mat = torch.Tensor(np.load(args.root_dir + 'alpha_m_mat.npy'))

	# Update sample_weights
	for i in range(boosting_iters):
		print("boosting"+str(i))
		if i >= already_boosted:
			sample_weights_new, alpha_m = \
			sample_models(boosting_iters=i, sample_weights=sample_weights_new, retrain_epoch=args.retrain_epochs)
			alpha_m_mat = torch.cat((alpha_m_mat, alpha_m), 1)
			np.save(args.root_dir+'alpha_m_mat',alpha_m_mat.numpy())
		else:
			sample_weights_new = torch.Tensor(np.load(args.root_dir + 'sample_weights_'+str(i)+'.npy'))
		print('%s %d-th Sample done !' % (str(datetime.datetime.utcnow()), i))
		index_weak_cls = index_weak_cls + 1

		print('-'*40)
		print('After '+str(i)+ ' Boosting:')
		print('Max weight:', torch.max(sample_weights_new))
		print('Min weight:', torch.min(sample_weights_new))
		print('*'*20+'  Ensembling  '+ '*'*20)
		get_ens_test(i+1)





