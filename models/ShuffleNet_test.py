import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time

# import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam,lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import transforms, datasets
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.distributed as dist 
import torch.utils.data.distributed
from torch.multiprocessing import Pool, Process
# define transformations for train
train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(p=.40),
	transforms.RandomRotation(30),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# define transformations for test
test_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	# implement mish activation function
def f_mish(input):
	'''
	Applies the mish function element-wise:
	mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
	'''
	return input * torch.tanh(F.softplus(input))

# implement class wrapper for mish activation function
class mish(nn.Module):
	'''
	Applies the mish function element-wise:
	mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

	Shape:
		- Input: (N, *) where * means, any number of additional
		  dimensions
		- Output: (N, *), same shape as the input

	Examples:
		>>> m = mish()
		>>> input = torch.randn(2)
		>>> output = m(input)

	'''
	def __init__(self):
		'''
		Init method.
		'''
		super().__init__()

	def forward(self, input):
		'''
		Forward pass of the function.
		'''
		return f_mish(input)

class BasicConv2d(nn.Module):

	def __init__(self, input_channels, output_channels, kernel_size, activation = 'relu', **kwargs):
		super().__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
		self.bn = nn.BatchNorm2d(output_channels)
		
		if (activation == 'relu'):
			self.relu = nn.ReLU(inplace=True)
			
		if (activation == 'swish'):
			self.relu = swish()
			
		if (activation == 'mish'):
			self.relu = mish()
	
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x

class ChannelShuffle(nn.Module):

	def __init__(self, groups):
		super().__init__()
		self.groups = groups
	
	def forward(self, x):
		batchsize, channels, height, width = x.data.size()
		channels_per_group = int(channels / self.groups)

		#"""suppose a convolutional layer with g groups whose output has
		#g x n channels; we first reshape the output channel dimension
		#into (g, n)"""
		x = x.view(batchsize, self.groups, channels_per_group, height, width)

		#"""transposing and then flattening it back as the input of next layer."""
		x = x.transpose(1, 2).contiguous()
		x = x.view(batchsize, -1, height, width)

		return x

class DepthwiseConv2d(nn.Module):

	def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
		super().__init__()
		self.depthwise = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
			nn.BatchNorm2d(output_channels)
		)

	def forward(self, x):
		return self.depthwise(x)

class PointwiseConv2d(nn.Module):
	def __init__(self, input_channels, output_channels, **kwargs):
		super().__init__()
		self.pointwise = nn.Sequential(
			nn.Conv2d(input_channels, output_channels, 1, **kwargs),
			nn.BatchNorm2d(output_channels)
		)
	
	def forward(self, x):
		return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

	def __init__(self, input_channels, output_channels, stage, stride, groups, activation = 'relu'):
		super().__init__()
		
		if (activation == 'relu'):
			f_activation = nn.ReLU(inplace=True)
			
		if (activation == 'swish'):
			f_activation = swish()
			
		if (activation == 'mish'):
			f_activation = mish()

		#"""Similar to [9], we set the number of bottleneck channels to 1/4 
		#of the output channels for each ShuffleNet unit."""
		self.bottlneck = nn.Sequential(
			PointwiseConv2d(
				input_channels, 
				int(output_channels / 4), 
				groups=groups
			),
			f_activation
		)

		#"""Note that for Stage 2, we do not apply group convolution on the first pointwise 
		#layer because the number of input channels is relatively small."""
		if stage == 2:
			self.bottlneck = nn.Sequential(
				PointwiseConv2d(
					input_channels, 
					int(output_channels / 4),
					groups=groups
				),
				f_activation
			)
		
		self.channel_shuffle = ChannelShuffle(groups)

		self.depthwise = DepthwiseConv2d(
			int(output_channels / 4), 
			int(output_channels / 4), 
			3, 
			groups=int(output_channels / 4), 
			stride=stride,
			padding=1
		)

		self.expand = PointwiseConv2d(
			int(output_channels / 4),
			output_channels,
			groups=groups
		)

		self.relu = f_activation
		self.fusion = self._add
		self.shortcut = nn.Sequential()

		#"""As for the case where ShuffleNet is applied with stride, 
		#we simply make two modifications (see Fig 2 (c)): 
		#(i) add a 3 × 3 average pooling on the shortcut path; 
		#(ii) replace the element-wise addition with channel concatenation, 
		#which makes it easy to enlarge channel dimension with little extra 
		#computation cost.
		if stride != 1 or input_channels != output_channels:
			self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

			self.expand = PointwiseConv2d(
				int(output_channels / 4),
				output_channels - input_channels,
				groups=groups
			)

			self.fusion = self._cat
	
	def _add(self, x, y):
		return torch.add(x, y)
	
	def _cat(self, x, y):
		return torch.cat([x, y], dim=1)

	def forward(self, x):
		shortcut = self.shortcut(x)

		shuffled = self.bottlneck(x)
		shuffled = self.channel_shuffle(shuffled)
		shuffled = self.depthwise(shuffled)
		shuffled = self.expand(shuffled)

		output = self.fusion(shortcut, shuffled)
		output = self.relu(output)

		return output

class ShuffleNet(nn.Module):

	def __init__(self, num_blocks, num_classes=10, groups=3, activation = 'relu'):
		super().__init__()

		if groups == 1:
			out_channels = [24, 144, 288, 567]
		elif groups == 2:
			out_channels = [24, 200, 400, 800]
		elif groups == 3:
			out_channels = [24, 240, 480, 960]
		elif groups == 4:
			out_channels = [24, 272, 544, 1088]
		elif groups == 8:
			out_channels = [24, 384, 768, 1536]

		self.conv1 = BasicConv2d(3, out_channels[0], 3, padding=1, stride=1, activation = activation)
		self.input_channels = out_channels[0]

		self.stage2 = self._make_stage(
			ShuffleNetUnit, 
			num_blocks[0], 
			out_channels[1], 
			stride=2, 
			stage=2,
			groups=groups,
			activation = activation
		)

		self.stage3 = self._make_stage(
			ShuffleNetUnit, 
			num_blocks[1], 
			out_channels[2], 
			stride=2,
			stage=3, 
			groups=groups,
			activation = activation
		)

		self.stage4 = self._make_stage(
			ShuffleNetUnit,
			num_blocks[2],
			out_channels[3],
			stride=2,
			stage=4,
			groups=groups,
			activation = activation
		)

		self.avg = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(out_channels[3], num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.avg(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

	def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups, activation = 'relu'):
		"""make shufflenet stage 

		Args:
			block: block type, shuffle unit
			out_channels: output depth channel number of this stage
			num_blocks: how many blocks per stage
			stride: the stride of the first block of this stage
			stage: stage index
			groups: group number of group convolution 
		Return:
			return a shuffle net stage
		"""
		strides = [stride] + [1] * (num_blocks - 1)

		stage = []

		for stride in strides:
			stage.append(
				block(
					self.input_channels, 
					output_channels, 
					stride=stride, 
					stage=stage, 
					groups=groups,
					activation = activation
				)
			)
			self.input_channels = output_channels

		return nn.Sequential(*stage)

def shufflenet(activation = 'relu'):
	return ShuffleNet([4, 8, 4], num_classes = 1000, activation = activation)

train_stats = pd.DataFrame(columns = ['Epoch', 'Time per epoch', 'Avg time per step', 'Train loss', 'Train accuracy', 'Train top-5 accuracy'])

class AverageMeter(object):
	"""Computes and stores the average and current value"""
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

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
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

def train(train_loader, model, criterion, optimizer, epoch):
	global train_stats
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):

		# measure data loading time
		data_time.update(time.time() - end)

		# Create non_blocking tensors for distributed training
		input = input.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)

		# compute output
		output = model(input)
		loss = criterion(output, target)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		# compute gradients in a backward pass
		optimizer.zero_grad()
		loss.backward()

		# Call step of optimizer to update model params
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

	print('Epoch: [{0}][{1}/{2}]\t'
		  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
		  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
		  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
		  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
		   epoch, i, len(train_loader), batch_time=batch_time,
		   loss=losses, top1=top1, top5=top5))
	train_stats = train_stats.append({'Epoch': epoch, 'Time per epoch':batch_time.val, 'Avg time per step': batch_time.avg, 'Train loss' : losses.val, 'Train accuracy': top1.val, 'Train top-5 accuracy':top5.val}, ignore_index=True)


def adjust_learning_rate(initial_lr, optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = initial_lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

test_stats = pd.DataFrame(columns = ['Time taken', 'Avg time per step', 'Test loss', 'Test accuracy', 'Test top-5 accuracy'])

def test(test_loader, model, criterion):
	global test_stats
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (images, target) in enumerate(test_loader):
			images = images.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

			# compute output
			output = model(images)
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0], images.size(0))
			top5.update(acc5[0], images.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   batch_time=batch_time, loss=losses, top1=top1, top5=top5))
			test_stats = test_stats.append({'Time taken':batch_time.val, 'Avg time per step': batch_time.avg, 'Test loss' : losses.val, 'Test accuracy': top1.val, 'Test top-5 accuracy':top5.val}, ignore_index=True)


	return top1.avg, top5.avg

traindir = os.getcwd() + '/train'
valdir = os.getcwd() + '/val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dist.init_process_group(backend="nccl", init_method="tcp://172.31.22.234:23456", rank=0, world_size=16)

train_dataset = datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

#train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_sampler = None
train_loader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
		num_workers=4, pin_memory=True, sampler=train_sampler)

testloader = DataLoader(
		datasets.ImageFolder(valdir,transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])), batch_size=batch_size, shuffle=False,
		num_workers=4, pin_memory=True)

print("-------------------Data Loaded------------------")
num_epochs = 30
learning_rate = 0.01
model = shufflenet(activation='mish')
# set loss function
criterion = nn.CrossEntropyLoss()
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	model = nn.DataParallel(model)

model.to(device)
# set optimizer, only train the classifier parameters, feature parameters are frozen
optimizer = Adam(model.parameters(), lr=learning_rate)

#train the model
steps = 0
running_loss = 0
epochs = 100
starting_lr = learning_rate
best_prec1 = 0
print("-------------------Model Ready------------------")

for epoch in range(num_epochs):
	# Set epoch count for DistributedSampler
	# train_sampler.set_epoch(epoch)

	# Adjust learning rate according to schedule
	adjust_learning_rate(starting_lr, optimizer, epoch)

	# train for one epoch
	print("\nBegin Training Epoch {}".format(epoch+1))
	train(train_loader, model, criterion, optimizer, epoch)
		
	print("Epoch Completed")
	if (epoch + 1) % 10 == 0:
		torch.save(model.state_dict, os.getcwd() + "/Trained_"+ str(epoch + 1)+".pth")

top1acc, top5acc = test(testloader, model, criterion)
print("Top-1 Test Accuracy : ", top1acc,"\tTop-5 Test Accuracy : ", top5acc)

train_stats.to_csv('train_log_ShuffleNet_Mish.csv')
test_stats.to_csv('test_log_ShuffleNet_Mish.csv')
