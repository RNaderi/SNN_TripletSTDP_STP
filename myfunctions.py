# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:52:33 2023

@author:  Naderi
"""
import numpy as np

#########################Parameters###############################
n_e=400
##################################################################
def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments



def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def test_acc_calculation(resultPopVecs,inputNumbers,assignments,testSamples):
    
    testing_ending = '200'
    start_time_testing = 0
    end_time_testing = int(testing_ending)
    n_e = 400                                         # Number of excitatory neurons
    n_input = 784                                     # Number of input neurons
    ending = ''
    testing_result_monitor=resultPopVecs
    testing_input_numbers=inputNumbers
    counter = 0 
    num_tests = end_time_testing //testSamples
    sum_accurracy = [0] * num_tests
    while (counter < num_tests):
        end_time = min(end_time_testing, 10000*(counter+1))
        start_time = 10000*counter
        test_results = np.zeros((10, end_time-start_time))
        print('calculate test accuracy')
        for i in range(end_time - start_time-1):
            test_results[:,i] = get_recognized_number_ranking(assignments,testing_result_monitor[i+start_time,:])
        difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
        correct = len(np.where(difference == 0)[0])
        incorrect = np.where(difference != 0)[0]
        sum_accurracy[counter] = correct/float(end_time-start_time) * 100
        test_acc=sum_accurracy[counter]
        counter += 1
        
    return test_acc
    
    
