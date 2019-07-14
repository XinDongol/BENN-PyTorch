
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch,cv2
import time
import datetime
import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/../../')
import data
import util
import numpy as np

from models import nin
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',
	help='set if only CPU is available')
parser.add_argument('--data', action='store', default='./data/',
	help='dataset path')
parser.add_argument('--arch', action='store', default='nin',
	help='the architecture for the network: nin')
parser.add_argument('--lr', action='store', default='0.001',
	help='the intial learning rate')
parser.add_argument('--epochs', action='store', default='0',
	help='fisrt train epochs',type=int)
parser.add_argument('--retrain_epochs', action='store', default='100',
	help='re-train epochs',type=int)
parser.add_argument('--save_name', action='store', default='first_model',
	help='save the first trained model',type=str)
parser.add_argument('--load_name', action='store', default='first_model',
	help='load pretrained model',type=str)
parser.add_argument('--root_dir', action='store', default='models_boostD_SB_seq/',
	help='root dir for different experiments',type=str)
parser.add_argument('--pretrained', action='store', default=None,
	help='the path to the pretrained model')
parser.add_argument('--evaluate', action='store_true',
	help='evaluate the model')
args = parser.parse_args()
print('==> Options:',args)

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)




# define classes
classes = ('plane', 'car', 'bird', 'cat',
	'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# prepare the data
if not os.path.isfile(args.data+'/train_data'):
	 # check the data path
	raise Exception\
	 ('Please assign the correct data path with --data <DATA_PATH>')

print('==> Building Dataset ...')
trainset = data.dataset(root=args.data, train=True)

#normal train loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
 	shuffle=False, num_workers=4)
#normal test loader
testset = data.dataset(root=args.data, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
 	shuffle=False, num_workers=4)


def save_state(model, best_acc, save_name, root_dir=args.root_dir):
	print('==> Saving model ...')
	new_state_dict = model.state_dict()
	state = {
	 'best_acc': best_acc,
	 'state_dict': new_state_dict,
	 }
	torch.save(state, root_dir + save_name +'.pth.tar')



def train(epoch, sample_weights=torch.Tensor(np.ones((50000,1))/50000.0)):
	adjust_learning_rate(optimizer, epoch)
	model.train()

	sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights[:,0].double(), 50000)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, \
		shuffle=False, num_workers=4, \
		sampler=sampler)
	for batch_idx, (data, target) in enumerate(trainloader):
		bin_op.binarization()
		data, target = Variable(data.cuda()), Variable(target.cuda())
		optimizer.zero_grad()
		output = model(data) 
		# backwarding
		loss = criterion(output, target)
		loss.backward()
		# restore weights
		bin_op.restore()
		bin_op.updateBinaryGradWeight()
		optimizer.step()
		if batch_idx % 100 == 0:
		 	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
		 		epoch, batch_idx * len(data), len(trainloader.dataset),
		 		100. * batch_idx / len(trainloader), loss.data[0],
		 		optimizer.param_groups[0]['lr']))
	return trainloader 

def test(save_name, best_acc, sample_weights=torch.Tensor(np.ones((50000,1))/50000.0)):
	#global best_acc
	model.eval()
	test_loss = 0
	correct = 0
	bin_op.binarization()

	sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights[:,0].double(), 50000)
	testloader_in = torch.utils.data.DataLoader(trainset, batch_size=1000,
 	shuffle=False, num_workers=4, sampler=sampler)

	for data, target in testloader_in:
		data, target = Variable(data.cuda()), Variable(target.cuda())

		output = model(data)
		test_loss += criterion(output, target).data[0]
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	bin_op.restore()
	acc = 100. * correct / len(testloader_in.dataset)

	if acc > best_acc:
		best_acc = acc
		save_state(model, best_acc, save_name)

	test_loss = test_loss / len(testloader_in.dataset) * 1000
	print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
		test_loss , correct, len(testloader_in.dataset),
		100. * correct / len(testloader_in.dataset)))
	print('Best Train Accuracy: {:.2f}%\n'.format(best_acc))
	return best_acc

def eval_test():
   #global best_acc
   model.eval()
   test_loss = 0
   correct = 0
   bin_op.binarization()
   for data, target in testloader:
       data, target = Variable(data.cuda()), Variable(target.cuda())
                                   
       output = model(data)
       test_loss += criterion(output, target).data[0]
       pred = output.data.max(1, keepdim=True)[1]
       correct += pred.eq(target.data.view_as(pred)).cpu().sum()
   bin_op.restore()
   acc = 100. * correct / len(testloader.dataset)
   test_loss /= len(testloader.dataset)
   print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
       test_loss * 1000, correct, len(testloader.dataset),
       100. * correct / len(testloader.dataset)))
   return





def adjust_learning_rate(optimizer, epoch):
	update_list = [120, 200, 240, 280]
	if epoch in update_list:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			return
def reset_learning_rate(optimizer, lr=0.001):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return



# prepare the options


# define the model
print('==> building model',args.arch,'...')
if args.arch == 'nin':
	model = nin.Net()
elif args.arch == 'allbinnet':
	model = nin.AllBinNet()
else:
	raise Exception(args.arch+' is currently not supported')

# initialize the model
if not args.pretrained:
	print('==> Initializing model parameters ...')
	
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			m.weight.data.normal_(0, 0.05)
			m.bias.data.zero_()
else:
	print('==> Load pretrained model form', args.pretrained, '...')
	pretrained_model = torch.load(args.pretrained)
	best_acc = pretrained_model['best_acc']
	model.load_state_dict(pretrained_model['state_dict'])

if not args.cpu:
	model.cuda()
	model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
print(model)

# define solver and criterion
base_lr = float(args.lr)
param_dict = dict(model.named_parameters())
params = []

for key, value in param_dict.items():
	params += [{'params':[value], 'lr': base_lr,
	'weight_decay':0.00001}]

	optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)

criterion = nn.CrossEntropyLoss()
criterion_seperated = nn.CrossEntropyLoss(reduce=False)
# define the binarization operator
bin_op = util.BinOp(model, 'nin')




def update_weights(softmax_output, target, sample_weights):
	print("start updating..")
	pred_numpy = softmax_output
	target_numpy = target
	pred_numpy = torch.squeeze(pred_numpy)
	miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
	miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
	miss = miss.unsqueeze(1)

	err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
	alpha_m = 0.1*(torch.log((1 - err_m) / float(err_m)) + torch.log(torch.Tensor([9])))

	prior_exp = torch.t(torch.Tensor(alpha_m * miss2))
	sample_weights_new = sample_weights * torch.exp(prior_exp)

	print("weights updated!")
	return sample_weights_new, alpha_m


def get_error_output(data, target, batch_sample_weights):

	data, target = Variable(data.cuda()), Variable(target.cuda())
	output = model(data)
	loss = (criterion_seperated(output, target)*Variable(batch_sample_weights.cuda().float())).mean()

	return output

def sample_models(boosting_iters, sample_weights, retrain_epoch=100):
	print(str(datetime.datetime.utcnow())+" Start boosting iter: "+str(boosting_iters) )
	print('===> Start retraining ...')

	best_acc = 0
	reset_learning_rate(optimizer, lr=0.001)
	for epoch in range(1, retrain_epoch+1):
		train(epoch, sample_weights)
		if epoch % 5 ==0:
			best_acc = test(str(boosting_iters), best_acc, sample_weights)
			eval_test()

	pretrained_model = torch.load(args.root_dir+ str(boosting_iters) +'.pth.tar')
	best_acc = pretrained_model['best_acc']
	model.load_state_dict(pretrained_model['state_dict'])

	model.eval()
	bin_op.binarization()

	pred_output = torch.Tensor(np.zeros((50000, 1))) #torch tensor in cpu
	label_in_tensor = torch.Tensor(np.zeros((50000, ))) #torch tensor in cpu

	for batch_idx, (data, target) in enumerate(trainloader):
		batch_size = target.size(0)
		batch_sample_weights = sample_weights[batch_idx*batch_size:(batch_idx+1)*batch_size]

		batch_softmax_output = get_error_output(data, target, batch_sample_weights)
		pred_output[batch_idx*batch_size:(batch_idx+1)*batch_size,:] = batch_softmax_output.max(1, keepdim=True)[1].data.cpu()
		label_in_tensor[batch_idx*batch_size:(batch_idx+1)*batch_size] = target

	bin_op.restore()
	best_sample_weights, best_alpha_m = update_weights(pred_output, label_in_tensor, sample_weights)

	return best_sample_weights, best_alpha_m

def use_sampled_model(sampled_model, data, target, batch_sample_weights):
	pretrained_model = torch.load(sampled_model)
	best_acc = pretrained_model['best_acc']
	model.load_state_dict(pretrained_model['state_dict'])

	model.eval()
	bin_op.binarization()
	return get_error_output(data, target, batch_sample_weights)


def combine_softmax_output(pred_test_i, pred_test, alpha_m_mat, i):
	pred_test_delta = alpha_m_mat[0][i] * pred_test_i
	pred_test = torch.add(pred_test, pred_test_delta.cpu())
	return pred_test

def most_common_element(pred_mat, alpha_m_mat, num_boost):
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(1000):
        best_value = -1000
        best_pred = -1
        for j in range(10):
            mask = [int(x) for x in (pred_mat[i,:] == j*np.ones((num_boost,), dtype=int))]
            if np.sum(mask * alpha_m_mat[0][0:0+num_boost]) > best_value:
                best_value = np.sum(mask * alpha_m_mat[0][0:0+num_boost])
                best_pred = j
        pred_most = np.append(pred_most, best_pred)

    return pred_most

def most_common_element_train(pred_mat, alpha_m_mat, num_boost):
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(128):
        best_value = -1000
        best_pred = -1
        for j in range(10):
            mask = [int(x) for x in (pred_mat[i,:] == j*np.ones((num_boost,), dtype=int))]
            if np.sum(mask * alpha_m_mat[0][0:0+num_boost]) > best_value:
                best_value = np.sum(mask * alpha_m_mat[0][0:0+num_boost])
                best_pred = j
        pred_most = np.append(pred_most, best_pred)

    return pred_most


if __name__ == '__main__':

	best_acc = 0
	if args.epochs!=0:
		for epoch in range(1, int(args.epochs)+1):
			train(epoch)
			best_acc = test(args.save_name, best_acc)
	else:
		print('==> Load pretrained model from ...')
		pretrained_model = torch.load(args.root_dir+args.load_name+'.pth.tar')
		model.load_state_dict(pretrained_model['state_dict'])

	boosting_iters = 32
	sample_weights_new = torch.Tensor(np.ones((50000,1)) / 50000.)

	index_weak_cls = 0;

	alpha_m_mat = torch.Tensor();

	# Update sample_weights
	for i in range(boosting_iters):
		print("boosting "+str(i))
		sample_weights_new, alpha_m = sample_models(boosting_iters=i, sample_weights=sample_weights_new, retrain_epoch=args.retrain_epochs)
		print('%s %d-th Sample done !' % (str(datetime.datetime.utcnow()), i))
		
		index_weak_cls = index_weak_cls + 1
		alpha_m_mat = torch.cat((alpha_m_mat, alpha_m), 1)

	print("boosting finished!")

	pred_store = torch.Tensor();

	#use the sampled model
	for num_boost in range(1,32):
		final_loss_total = []
		final_loss_total_2 = []
		for data, target in testloader:
			pred_store = torch.Tensor();
			pred_test = torch.Tensor(np.zeros((1000,10)))
			for i in range(num_boost):
				pred_test_i = use_sampled_model(args.root_dir+ str(i) +'.pth.tar', data, target, torch.Tensor(np.ones((1000,1))/1.0))
				pred = pred_test_i.max(1, keepdim=True)[1]
				pred_store = torch.cat((pred_store, pred.data.cpu().float()), 1)

				pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, alpha_m_mat, i)

			pred_most = most_common_element(pred_store.numpy(), alpha_m_mat.numpy(), num_boost)
			pred_most = torch.Tensor(pred_most)
			pred_test = pred_test.max(1, keepdim=True)[1]
			pred_test = pred_test.squeeze(1)

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

        os._exit(0)

