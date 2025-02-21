""" Created by A.Rezaei"""
""" code to calculate NCS for all pipelines and datasets"""


"''''''''''''''''''''''''''''''''''''''''''''' Imports '''''''''''''''''''''''''''''''''''''''''''''''''''"

import numpy as np
import pandas as pd
import math
import pickle 
import os
from tqdm import tqdm

"''''''''''''''''''''''''''''''''''''''''''''' Settings ''''''''''''''''''''''''''''''''''''''''''''''''''"

main_dir = r'.\visualization data'
save_dir = r'{}\SAM\50_time_steps_final1'.format(main_dir)

pipelines = ['pipe0', 'pipe1', 'pipe2']
datasets =  ['Digit', 'DigitNoisy', 'Letter', 'LetterNoisy']

save_tscs_time_series = False
save_ncs = True

gamma = 0.2

max_t = 110000 # length of the time series

# we calculate ncs for times in this list
number_steps = 50
interval = int(max_t/(number_steps))
ncs_ts = list(range(0, max_t + 1, interval))

"''''''''''''''''''''''''''''''''''''''''''''' Functions '''''''''''''''''''''''''''''''''''''''''''''''''"

def create_neuron_spike_times(neuron_index, neurons_spike_times):
    """
    creates a dictionary
    keys -> neuron indices
    values -> spike times of the key neuron

    """
    neuron_spike_times = {}
    for n, s in zip(neuron_index, neurons_spike_times):
        if n in neuron_spike_times.keys():
            neuron_spike_times[n].append(s)
        else:
            neuron_spike_times[n] = [s]
    return neuron_spike_times

def calculate_tscs(neuron_spike_times, gamma=0.2):
    """
    returns a dict
    keys -> neuron indices
    values -> the tscs value for each spike time of the key neuron

    """
    neurons_tscs = {}
    # we compute TSCS for all spike times
    for n, sts in neuron_spike_times.items():
        tscs_tmp = []
        for t in sts:
            # sort all spike times of each neuron
            spike_times = sorted(sts)
            # get the index of the current spike time to find the previous one
            t_idx = spike_times.index(t)
            if t_idx > 0:
                # t_prime is the previous spike time in the formula
                t_prime = spike_times[t_idx - 1]
            else:
                t_prime = 0
            # T(t,t') = exp(-gamma|t-t'|)
            T = math.exp(-gamma*abs(t - t_prime))
            tscs_tmp.append(T)
            
        neurons_tscs[n] = tscs_tmp
        
    return neurons_tscs

def create_time_series(data, times, max_t=108000):
    """
    given an array of values(spikes or tscs) and time of each value
    creates a time series from 0 ms to max_t ms

    """
    time_series = {}
    
    for item, t in zip(data.items(), times.values()):
        n, values = item
        ts = np.zeros(max_t)
        # ts[ts==0] = np.nan
        
        ts[t] = values
        
        time_series[n] = ts
        
    return time_series

def save_pickle(obj, path):
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def read_pickle(path):
    
    return pd.read_pickle(path)
        
def ncs(tscs_time_series, ncs_ts):
    """
    given the tscs time series for all neurons (dict), and the time steps in
    which ncs must be calculated, it returns a dict as follows:
        keys -> time steps
        values -> a dict containing neuron indices as keys and ncs values as values

    """
    ncs_dict = {}
    
    for t in ncs_ts:
        neurons_ncs = {}
        for n, tsc_ts in tscs_time_series.items():
            # this is p in the formula
            pre_times = tsc_ts[0:t]
            
            # N = sigma T(t,t')
            # here t is one of the times choosen in ncs_ts 
            # t' is the previous spike time of t which here
            # we sum all the tscs from the all times before t
            # if t in neuron_spike_times[n]:
            #     ncs = sum(pre_times)
            # else:
            #     ncs = 0
            
            ncs = sum(pre_times)
            neurons_ncs[n] = ncs
            
        ncs_dict[t] = neurons_ncs
        
    return ncs_dict        

"''''''''''''''''''''''''''''''''''''''''''''' Main ''''''''''''''''''''''''''''''''''''''''''''''''''''''"

if not os.path.exists(save_dir):
   os.makedirs(save_dir)
   
for dataset in tqdm(datasets):  
    
    for pipeline in pipelines:
        
        spike_time_file = r'{}\{}\{}\raster_time_{}_{}.npy'.format(main_dir,
                                                                                    dataset,
                                                                                    pipeline,
                                                                                    pipeline,
                                                                                    dataset)
        neuron_index_file =  r'{}\{}\{}\raster_neuronIndex_{}_{}.npy'.format(main_dir,
                                                                                                dataset,
                                                                                                pipeline,
                                                                                                pipeline,
                                                                                                dataset)
        
        print(spike_time_file)                                                                    
        if not os.path.isfile(spike_time_file):
            continue
            
        neurons_spike_times = np.load(spike_time_file)
        neurons_spike_times = neurons_spike_times.astype(int)

        neuron_index = np.load(neuron_index_file)
        
        neuron_spike_times = create_neuron_spike_times(neuron_index,
                                                       neurons_spike_times)   
        
         
        neurons_tscs = calculate_tscs(neuron_spike_times, gamma=gamma)
        
        tscs_time_series = create_time_series(neurons_tscs,
                                              neuron_spike_times, max_t=max_t)
        
        
        ncs_dict = ncs(tscs_time_series, ncs_ts) 
        
        if save_tscs_time_series:
            save_pickle(tscs_time_series, r'{}\tscs_{}_{}.pickle'.format(save_dir,
                                                                         dataset,
                                                                         pipeline))
            
        if save_ncs:
            save_pickle(ncs_dict, r'{}\ncs_{}_{}.pickle'.format(save_dir,
                                                                dataset,
                                                                pipeline))