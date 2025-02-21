'''
Reimplemented by R.Naderi
'''

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import os
import scipy 
from brian2 import *
import collections
import matplotlib.pyplot as plt
from functions.data import get_labeled_data


#------------------------------------------------------------------------------ 
# parameters
#------------------------------------------------------------------------------     

MNIST_data_path = './dataset/mnist-digit/'
data_path = './activity/'
#training_ending = '800'
testing_ending = '200'
#start_time_training = 0
#end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)
n_e = 400                                         # Number of excitatory neurons
n_input = 784                                     # Number of input neurons
ending = ''

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments


testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + 'noSTP.npy')
testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + 'noSTP.npy')
print('get assignments')
assignments =np.load(data_path + 'assignment400.npy')

#--#########################################################
#accuracy calculation
#----------------------------------------------
counter = 0 
num_tests = end_time_testing // 200
sum_accurracy = [0] * num_tests
while (counter < num_tests):
    end_time = min(end_time_testing, 10000*(counter+1))
    start_time = 10000*counter
    test_results = np.zeros((10, end_time-start_time))
    print('calculate accuracy for sum')
    for i in range(end_time - start_time-1):
        test_results[:,i] = get_recognized_number_ranking(assignments, 
                                                          testing_result_monitor[i+start_time,:])
    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct/float(end_time-start_time) * 100
    print('accuracy: ', sum_accurracy[counter], ' num incorrect: ', len(incorrect))
    counter += 1