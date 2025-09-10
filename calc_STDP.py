"""
	PyTorch implementatation
	of spike-timing-dependent plasticity
	
	Kevin Max, OIST, 2025

	https://github.com/kma-code/PyTorch_STDP
"""


import math
import torch
import torch.nn.functional as F
import torchaudio.functional as F2
import numpy as np
import matplotlib.pyplot as plt

def calc_dW_from_spike_trains(pre_spike_train, post_spike_train, causal_STDP_kernel=None, anticausal_STDP_kernel=None):
	"""
		Calculates STDP weight update given pre- and post-synaptic spike trains
		and STDP kernels for causal (pre before post)
		and anti-causal (post before pre) spike pairs.
		Calculates all spike correlations (not just nearest neighbors).

		expects:
			pre_spike_train: torch.Tensor of size (batch_size, n_pre, n_time_steps)
			post_spike_train: torch.Tensor of size (batch_size, n_post, n_time_steps)
			causal_STDP_kernel: torch.Tensor of size (kernel_length)
								will be applied as convolutional kernel along time axis
			anticausal_STDP_kernel: torch.Tensor of size (kernel_length)
									will be applied as convolutional kernel along time axis
		returns:
			array dW of shape (n_post,n_pre)
	"""

	batch_size = pre_spike_train.shape[0]
	n_pre = pre_spike_train.shape[1]
	n_post = post_spike_train.shape[1]
	len_spike_trains = pre_spike_train.shape[-1]

	# expand kernel to batch size and post-synaptic neurons
	pre_kernel = causal_STDP_kernel.repeat((batch_size, n_pre, 1))
	# convolve the pre-synaptic spikes with the kernel along the time axis
	pre_tracker = F2.convolve(pre_spike_train, pre_kernel)[:,:,:len_spike_trains]
	assert pre_spike_train.shape == pre_tracker.shape

	# expand kernel to batch size and post-synaptic neurons
	post_kernel = anticausal_STDP_kernel.repeat((batch_size, n_post, 1))
	# convolve the post-synaptic spikes with the kernel along the time axis
	post_tracker = F2.convolve(post_spike_train, post_kernel)[:,:,:len_spike_trains]
	assert post_spike_train.shape == post_tracker.shape

	# multiply tracker with spike trains and sum along batches
	dW = torch.matmul(post_spike_train, pre_tracker.permute([0,2,1])).sum(axis=0)
	dW += torch.matmul(post_tracker, pre_spike_train.permute([0,2,1])).sum(axis=0)

	return dW

def exp_kernel(tau=5.0, a=1.0, bias=0.0):
	"""
		Defines a general exponential kernel
		with decay constant tau
		and kernel width of 5 tau
	"""
	kernel = a * torch.exp(-torch.arange(5*tau)/tau) + bias
	# no change for spikes arriving at exactly the same time
	kernel[0] = 0.0
	return kernel

def box_kernel(tau=5.0, a=1.0, bias=0.0):
	"""
		Defines a general box kernel of width tau
	"""
	kernel = a * torch.ones(int(tau)) + bias
	kernel[0] = 0.0
	return kernel

if __name__ == '__main__':

	batch_size = 2
	n_pre = 3 			# number of pre-synaptic neurons
	n_post = 4			# number of post-synaptic neurons
	n_time_steps = 1000

	print(f"Generating spike trains for {n_pre} pre- and {n_post} post-synaptic neurons with batch size {batch_size} for {n_time_steps} steps.")
	pre_spike_train = (torch.rand(size=(batch_size, n_pre, n_time_steps)) < 0.25).float() 		# spike train with p = 0.25
	post_spike_train = (torch.rand(size=(batch_size, n_post, n_time_steps)) < 0.25).float()

	# define STDP kernels
	causal_STDP_kernel = exp_kernel(tau=5.0, a=1.0) 			# tau = 5 means that the exponential decay constant is 5 time steps;
																# a = 1 for LTP of causal spike pairs
	anticausal_STDP_kernel = exp_kernel(tau=5.0, a=-1.0) 		# a = -1 is for LTD of anti-causal spike pairs


	
	# plot spikes
	fig, axes = plt.subplots(2, sharex=True, figsize=(11, 8))
	for ax, data in zip(axes, [pre_spike_train, post_spike_train]):
	    data = data.view(n_time_steps, -1).nonzero()
	    ax.scatter(data[:,0], data[:,1], marker='.', c='black')
	plt.tight_layout()
	plt.show()

	with torch.no_grad():
		print("Calculating STDP weight update connecting pre- to post-neurons")
		dW = calc_dW_from_spike_trains(pre_spike_train=pre_spike_train,
								post_spike_train=post_spike_train,
								causal_STDP_kernel=causal_STDP_kernel,
								anticausal_STDP_kernel=anticausal_STDP_kernel)
		torch.set_printoptions(precision=3, linewidth=1000)
		print("Weight update:")
		print(dW)



