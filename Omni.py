import torch, os
from torch import optim
from torch.autograd import Variable
import numpy as np
from Omni_NaiveRN import NaiveRN
from omniglotNShot import OmniglotNShot
from torch.optim import lr_scheduler
import argparse

import scipy.stats, sys


def mean_confidence_interval(accs, confidence = 0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf( (1 + confidence) / 2, n - 1)
    return m, h


# save best acc info, to save the best model to ckpt.
best_accuracy = 0
def evaluation(net, batchsz, n_way, k_shot, imgsz, episodesz, threhold, mdl_file):
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
			pred, correct = net(support_x, support_y, query_x_mini.contiguous(), query_y_mini, False)
			correct = correct.sum() # multi-gpu
			# pred: [b, nway]
			preds.append(pred)
			total_correct += correct.data[0]
			total_num += query_y_mini.size(0) * query_y_mini.size(1)
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

	return accuracy, sem


def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-n', help='n way')
	argparser.add_argument('-k', help='k shot')
	argparser.add_argument('-b', help='batch size')
	argparser.add_argument('-l', help='learning rate', default=1e-3)
	argparser.add_argument('-t', help='threshold to test all episodes', default=0.97)
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	k_query = 1
	batchsz = int(args.b)
	imgsz = 84
	lr = float(args.l)
	threshold = float(args.t)

	db = OmniglotNShot('dataset', batchsz=batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=imgsz)
	print('Omniglot: no rotate!  %d-way %d-shot  lr:%f' % (n_way, k_shot, lr))
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
			accuracy, _ = 0,0 # evaluation(net, batchsz, n_way, k_shot, imgsz, 300, threshold, mdl_file)
			scheduler.step(accuracy)



		# 2. train
		support_x, support_y, query_x, query_y = db.get_batch('train')
		support_x = Variable(torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		query_x = Variable(torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		support_y = Variable(torch.from_numpy(support_y).int()).cuda()
		query_y = Variable(torch.from_numpy(query_y).int()).cuda()

		loss = net(support_x, support_y, query_x,  query_y)
		total_train_loss += loss.data[0]

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# 3. print
		if step % 20 == 0 and step != 0:
			print('%d-way %d-shot %d batch> step:%d, loss:%f' % (
				n_way, k_shot, batchsz, step, total_train_loss))
			total_train_loss = 0
















if __name__ == '__main__':
	main()
