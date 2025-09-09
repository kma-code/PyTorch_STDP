# snnTorch STDP: Efficient implementation of spike-timing-dependent plasticity in snnTorch/PyTorch

I was surprised that I couldn't find an efficient implementation of STDP for use in snnTorch anywhere. So here it is: a fast PyTorch implementation for offline STDP.

## How to install


```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## How to run example


```
python calc_STDP.py
```

## What it does

The function `calc_dW_from_spike_trains` implements the idea of implementing STDP via spike-tracking from the Neuromatch Academy [Bonus Tutorial: Spike-timing dependent plasticity (STDP)](https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial4.html#keeping-track-of-pre-and-postsynaptic-spikes) (Content creators: Qinglong Gu, Songtin Li, John Murray, Richard Naud, Arvind Kumar).

The relevant code snippet is this:
https://github.com/kma-code/snnTorch_STDP/blob/2bd9360d5838f39cd18f6b48727be7ea97324dda/calc_STDP.py#L18

This implementation is **offline/batched**, meaning that STDP is not calculated during every time step (online), but across a recorded time series of `n_time_steps` steps (e.g. a stimulus presentation).

Comes with two STDP kernels for **free**: `exp_kernel` and `box_kernel`.
