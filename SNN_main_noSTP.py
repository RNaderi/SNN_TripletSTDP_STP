'''
Reimplemented by R.Naderi
'''
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
from os import chdir,getcwd
import scipy
from brian2 import *
import os
import brian2 as b2
from brian2tools import *
import datetime
import myfunctions
prefs.codegen.target = 'auto'                            # use the Python fallback

#------------------------------------------------------------------------------
#Parameters for initialization
#------------------------------------------------------------------------------
MNIST_data_path = './dataset/mnist-digit/'               # specify the location of the MNIST data-for testing and training
current_dir='K:\\SNN_TripletSTDP_STP'
os.chdir(current_dir)
data_path = './' 

trained_data_path='TrainedData2/mnist-digit/'           # specify the location of trained data(weight, theta,assignment, spike count for exc neurons per image sample and performance)
test_mode = False                                        # Change this to False to retrain the network
classes_input=''
ending = ''
updateInterval=200
testSamples=200
Epochs=15
trainSamples=800*Epochs
#trainSamples=2
saveConnectionsInterval=800
fig_num = 1
n_input =784                                           #input layer neurons(28*28)
n_e = 400                                              #excitatory neurons
n_i = n_e                                              #Inhibitory neurons
single_example_time =   0.35 * b2.second               #350 ms
resting_time = 0.15 * b2.second                        #150 ms
# number of classes to learn
if classes_input == '':
    classes = range(10)
else:
    classes = set([ int(token) for token in classes_input.split(',') ])

if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = testSamples
    do_plot_performance = False
    ee_STDP_on = False
    update_interval = updateInterval
else:
    weight_path = data_path + 'random/'
    num_examples = trainSamples
    do_plot_performance = True
    ee_STDP_on = True
    update_interval = updateInterval
    
runtime = num_examples * (single_example_time + resting_time)
    
if num_examples <= 60000:
    save_connections_interval = saveConnectionsInterval
else:
    save_connections_interval = 10000
    
assignments = np.zeros(n_e)                              #assigned labels to exc neurons
input_numbers = [0] * num_examples                       #original class
outputNumbers = np.zeros((num_examples, 10))             #predicted class
result_monitor = np.zeros((updateInterval,n_e))          #save spikeCount per sample for each exc neuron
#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print('save connections')
    for connName in save_conns:
        #print("connName",connName)
        conn = connections[connName]
        #print(conn.i.shape,conn.i[:].shape,conn.i[0],conn.j[0],conn.w[0])
        
        #connListSparse = zip(conn.i, conn.j, conn.w)
        connListSparse=np.zeros([n_input*n_e,3])
        connListSparse[:,0]=conn.i[:]
        connListSparse[:,1]=conn.j[:]
        connListSparse[:,2]=conn.w[:]
        #print(connListSparse[0,:],connListSparse[1,:])
        #print(connListSparse.shape)
        np.save(data_path + 'weights/' + connName + ending, connListSparse)
        np.save(data_path + trained_data_path+ connName + ending, connListSparse)

def save_theta(ending = ''):
    print('save theta')
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)
        #np.save(data_path + 'assignment/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)
        np.save(data_path + trained_data_path + 'theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)
        
def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            len_source = len(connections[connName].source)
            len_target = len(connections[connName].target)
            connection = np.zeros((len_source, len_target))
            connection[connections[connName].i, connections[connName].j] = connections[connName].w
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in range(n_e):#
                temp_conn[:,j] *= colFactors[j]
            connections[connName].w = temp_conn[connections[connName].i, connections[connName].j]

def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = np.zeros((n_input, n_e))
    connMatrix[connections[name].i, connections[name].j] = connections[name].w
    weight_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b2.figure(fig_num, figsize = (18, 18))
    im2 = b2.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b2.colorbar(im2)
    b2.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    print("correct "+str(current_example_num),correct)
    performance[current_evaluation] = correct / float(update_interval) * 100
    print("get_current_performance"+str(current_example_num),performance)
    return performance
def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    #time_steps = range(0, num_evaluations)
    time_steps = range(0, num_evaluations+1)
    #performance = np.zeros(num_evaluations)
    performance = np.zeros(num_evaluations+1)
    fig = b2.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b2.ylim(ymax = 100)
    b2.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

begin_time = datetime.datetime.now()

#------------------------------------------------------------------------------
# load DataSet
#------------------------------------------------------------------------------
X_train=np.load(MNIST_data_path +'training.npy')
y_train=np.load(MNIST_data_path +'trainingLabels.npy')
training = {'x': X_train, 'y': y_train, 'rows': 28, 'cols': 28}

X_test=np.load(MNIST_data_path+'testing.npy')
y_test=np.load(MNIST_data_path+'testingLabels.npy')
testing = {'x': X_test, 'y': y_test, 'rows': 28, 'cols': 28}
#------------------------------------------------------------------------------
# set model parameters and equations
#------------------------------------------------------------------------------
np.random.seed(0)   
v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b2.ms,10*b2.ms)
delay['ei_input'] = (0*b2.ms,5*b2.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 40*b2.ms
nu_ee_pre =  0.0001                              # learning rate
nu_ee_post = 0.01                                # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b2.ms
    theta_plus_e = 0.05 * b2.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b2.mV
v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

b2.ion()

neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}


neuron_groups['e'] = b2.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e_str, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['i'] = b2.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i_str, refractory= refrac_i, reset= v_reset_i_str, method='euler')

#------------------------------------------------------------------------------
# create network population and recurrent connections :excitaory to inhibitory and vice versa
#------------------------------------------------------------------------------
for subgroup_n, name in enumerate(population_names):
    #print('create neuron group', name)

    neuron_groups[name+'e'] = neuron_groups['e'][subgroup_n*n_e:(subgroup_n+1)*n_e]
    neuron_groups[name+'i'] = neuron_groups['i'][subgroup_n*n_i:(subgroup_n+1)*n_e]

    neuron_groups[name+'e'].v = v_rest_e - 40. * b2.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b2.mV
    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy') * b2.volt
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b2.mV

    print('create recurrent connections')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + ending + '.npy')
        model = 'w : 1'
        pre = 'g%s_post += w' % conn_type[0]
        post = ''
        
        connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        connections[connName].connect(True) # all-to-all connection
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

        spike_counters[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])
#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
for name in input_population_names:                   #  X
    input_groups[name+'e'] = b2.PoissonGroup(n_input, 0*Hz)
    print(input_groups[name+'e'])
    spike_monitors[name+'e'] = b2.SpikeMonitor(input_groups[name+'e'])

    
for name in input_connection_names:                   #['XA']
    print('create connections between', name[0], 'and', name[1]) # X , A
    for connType in input_conn_names:  #ee_input
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        #print(weightMatrix.shape)
             
           
        ######################################################################     
        model = 'w : 1'
        pre = 'g%s_post += w' % connType[0]
        post = ''
        if ee_STDP_on:
            #print('create STDP for connection', name[0]+'e'+name[1]+'e')
            model += eqs_stdp_ee
            pre += '; ' + eqs_stdp_pre_ee
            post = eqs_stdp_post_ee

        connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        minDelay = delay[connType][0]
        maxDelay = delay[connType][1]
        deltaDelay = maxDelay - minDelay
        # TODO: test this
        connections[connName].connect(True) # all-to-all connection
        connections[connName].delay = 'minDelay + rand() * deltaDelay'
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------
spike_info=np.zeros([num_examples,n_e])
start = time.time()
#count the number of inputs whose sum(current_spike_count) < 5
sumCurrentSpikeCount=0

net = Network()
for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
        spike_monitors, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)

if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
    #print("performance",performance,performance_monitor) #0
for name in input_population_names:
    input_groups[name+'e'].rates = 0 * Hz
net.run(0*second)
j = 0
while j < (int(num_examples)):
    if test_mode:
        spike_rates = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        spike_rates = training['x'][j%800,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rates = spike_rates * Hz
    
    net.run(single_example_time)
    
    if j % update_interval == 0 and j > 0:
        assignments = myfunctions.get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
        
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))
        np.save(data_path + trained_data_path + 'assignment' + str(j), assignments)

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    spike_info[j,:]=current_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    
    if np.sum(current_spike_count) < 5:
        sumCurrentSpikeCount+=1
        #print(' this is number' ,sumCurrentSpikeCount," for sum(current_spike_count) < 5")
        input_intensity += 1
        for name in input_population_names:
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode:
            
            input_numbers[j] = testing['y'][j%10000]
        else:
            input_numbers[j] = training['y'][j%800]
        outputNumbers[j,:] = myfunctions.get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        if j % 800 == 0 and j > 0:
            print('epoch: ', j/800)
       
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                
                unused, performance =update_performance_plot(performance_monitor, performance, j, fig_performance)
                print('Classification performance', performance[:int(j/float(update_interval))+1])
                
        if j % save_connections_interval == 0 and j > 0 and not test_mode:
           np.save(data_path + trained_data_path + 'performance' + str(j), performance)
           np.save(data_path + trained_data_path + 'resultPopVecs' + str(j), result_monitor)
           
           
        for name in input_population_names:
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1
        #print("j comparison with update_interval",j)
        
print("sumCurrentSpikeCount",sumCurrentSpikeCount)
end = time.time()
print(f"Duration :  { end-start } seconds")
print(f"Duration :  { (end-start)/3600 } hours")
#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print('save results')

if not test_mode:
    save_theta()
    save_connections()
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/assignment' + str(n_e), assignments)
    #print(assignmnets)
    np.save(data_path + trained_data_path + 'performance' , performance)
        
    np.save(data_path + trained_data_path + 'resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + trained_data_path + 'assignment' + str(n_e), assignments)
    #np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples)+str("noSTP"), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples)+str("noSTP"), input_numbers)
    np.save(data_path + 'activity/spike_info' + str(num_examples)+str("noSTP"), spike_info)
    #np.save(data_path+'activity/raster_neuronIdx_Emnist_10class_pipe0',spike_monitors['Ae'].i)
    #np.save(data_path+'activity/raster_time_Emnist_10class_pipe0',spike_monitors['Ae'].t/b2.ms)
    #np.save(data_path+'predicted_Emnist_10class_pipe0',test_results[0,:])
    #np.save(data_path+'original_Emnist_10class_pipe0',testing_input_numbers[0:200])

