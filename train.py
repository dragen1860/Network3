import torch, os
from torch import optim
from torch.autograd import Variable
import numpy as np
from naive import NaiveRN
from omniglotNShot import OmniglotNShot
from torch.optim import lr_scheduler
import argparse

import scipy.stats, sys
from tensorboardX import SummaryWriter
import pickle

global_train_acc_buff = 0
global_train_loss_buff = 0
global_test_acc_buff = 0
global_test_loss_buff = 0
global_buff = []

def write2file(n_way, k_shot):
	global_buff.append([global_train_loss_buff, global_train_acc_buff, global_test_loss_buff, global_test_acc_buff])
	with open("omni%d%d.pkl"%(n_way, k_shot), "wb") as fp:  
		pickle.dump(global_buff, fp)


def mean_confidence_interval(accs, confidence = 0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf( (1 + confidence) / 2, n - 1)
    return m, h


# save best acc info, to save the best model to ckpt.
best_accuracy = 0
def evaluation(net, batchsz, n_way, k_shot, imgsz, episodesz, threhold, mdl_file, tb):
	"""
	obey the expriment setting of MAML and Learning2Compare, we randomly sample 600 episodes and 15 query images per query
	set.
	:param net:
	:param batchsz:
	:return:
	"""
	k_query = 15
	db = OmniglotNShot('dataset', batchsz=batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=imgsz)

	accs = []
	episode_num = 0 # record tested num of episodes

	total_loss = 0
	for i in range( 600 // batchsz):
		support_x, support_y, query_x, query_y = db.get_batch('test')
		support_x = Variable(torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		query_x = Variable(torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		support_y = Variable(torch.from_numpy(support_y).int()).cuda()
		query_y = Variable(torch.from_numpy(query_y).int()).cuda()

		# we will split query set into 15 splits.
		# query_x : [batch, 15*way, c_, h, w]
		# query_x_b : tuple, 15 * [b, way, c_, h, w]
		query_x_b = torch.chunk(query_x, k_query, dim= 1)
		# query_y : [batch, 15*way]
		# query_y_b: 15* [b, way]
		query_y_b = torch.chunk(query_y, k_query, dim= 1)
		preds = []
		net.eval()
		# we don't need the total acc on 600 episodes, but we need the acc per sets of 15*nway setsz.
		total_correct = 0
		total_num = 0
		for query_x_mini, query_y_mini in zip(query_x_b, query_y_b):
			# print('query_x_mini', query_x_mini.size(), 'query_y_mini', query_y_mini.size())
			pred, correct, loss = net(support_x, support_y, query_x_mini.contiguous(), query_y_mini, False)
			correct = correct.sum() # multi-gpu
			# pred: [b, nway]
			preds.append(pred)
			total_correct += correct.data[0]
			total_num += query_y_mini.size(0) * query_y_mini.size(1)

			total_loss += loss.data[0] / (n_way*k_query)
		# # 15 * [b, nway] => [b, 15*nway]
		# preds = torch.cat(preds, dim= 1)
		acc = total_correct / total_num
		print('%.5f,'%acc, end=' ')
		sys.stdout.flush()
		accs.append(acc)

		# update tested episode number
		episode_num += query_y.size(0)
		if episode_num > episodesz:
			# test current tested episodes acc.
			acc = np.array(accs).mean()
			if acc >= threhold:
				# if current acc is very high, we conduct all 600 episodes testing.
				continue
			else:
				# current acc is low, just conduct `episodesz` num of episodes.
				break


	# compute the distribution of 600/episodesz episodes acc.
	global best_accuracy
	accs = np.array(accs)
	accuracy, sem = mean_confidence_interval(accs)
	print('\naccuracy:', accuracy, 'sem:', sem)
	print('<<<<<<<<< accuracy:', accuracy, 'best accuracy:', best_accuracy, '>>>>>>>>')

	if accuracy > best_accuracy:
		best_accuracy = accuracy
		torch.save(net.state_dict(), mdl_file)
		print('Saved to checkpoint:', mdl_file)


	total_loss = total_loss / episode_num / k_query
	tb.add_scalar('test-acc', accuracy)
	tb.add_scalar('test-loss', total_loss)

	global global_test_loss_buff, global_test_acc_buff
	global_test_loss_buff = total_loss
	global_test_acc_buff = accuracy
	write2file(n_way, k_shot)


	return accuracy, sem


def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-n', help='n way', default=5)
	argparser.add_argument('-k', help='k shot', default=1)
	argparser.add_argument('-b', help='batch size', default=1)
	argparser.add_argument('-l', help='learning rate', default=1e-3)
	argparser.add_argument('-t', help='threshold to test all episodes', default=1)
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	k_query = 1
	batchsz = int(args.b)
	imgsz = 84
	lr = float(args.l)
	threshold = float(args.t)
	if threshold == 1:
		if n_way == 5 and k_shot == 1:
			threshold = 0.996
		elif n_way == 5 and k_shot == 5:
			threshold = 0.998
		elif n_way == 20 and k_shot == 1:
			threshold = 0.976
		elif n_way == 20 and k_shot == 5:
			threshold = 0.991

	tb = SummaryWriter('runs')
			
	db = OmniglotNShot('dataset', batchsz=batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=imgsz)
	print('Omniglot: rotate!  %d-way %d-shot  lr:%f, threshold:%f' % (n_way, k_shot, lr, threshold))
	net = NaiveRN(n_way, k_shot, imgsz).cuda()
	print(net)
	mdl_file = 'ckpt/omni%d%d.mdl'%(n_way, k_shot)

	if os.path.exists(mdl_file):
		print('recover from state: ', mdl_file)
		net.load_state_dict(torch.load(mdl_file))
	else:
		print('training from scratch.')

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total params:', params)

	input, input_y, query, query_y = db.get_batch('train')  # (batch, n_way*k_shot, img)
	print('get batch:', input.shape, query.shape, input_y.shape, query_y.shape)

	optimizer = optim.Adam(net.parameters(), lr=lr)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=15, verbose=True)

	total_train_loss = 0
	for step in range(100000000):

		# 1. test
		if step % 400 == 0:
			accuracy, _ = evaluation(net, batchsz, n_way, k_shot, imgsz, 100, threshold, mdl_file, tb)
			scheduler.step(accuracy)

		# 2. train
		support_x, support_y, query_x, query_y = db.get_batch('train')
		support_x = Variable(torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		query_x = Variable(torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		support_y = Variable(torch.from_numpy(support_y).int()).cuda()
		query_y = Variable(torch.from_numpy(query_y).int()).cuda()

		loss, correct = net(support_x, support_y, query_x,  query_y) 
		total_train_loss += loss.data[0]

		optimizer.zero_grad()
		loss.backward()
		# torch.nn.utils.clip_grad_norm(net.parameters(), 10)
		optimizer.step()

		# 3. print
		if step % 40 == 0 and step != 0:
			print('%d-way %d-shot %d batch> step:%d, loss:%f' % (
				n_way, k_shot, batchsz, step, total_train_loss))
			total_train_loss = 0
			acc = correct.data[0] / support_y.size(0) / support_y.size(1)
			tb.add_scalar('train-acc', acc)
			tb.add_scalar('train-loss', loss.data[0] / (n_way*k_shot))
			
			global global_train_loss_buff, global_train_acc_buff
			global_train_loss_buff = loss.data[0]/ (n_way*k_shot)
			global_train_acc_buff = acc
			write2file(n_way, k_shot)















if __name__ == '__main__':
	main()
