''' @author: A.Rezaei '''
""" Code to plot :
    Raster plot and spike activity
    NCS
    labeling distribution before and after the Update labeling phase
    confusion matrix
    model performances"""

"''''''''''''''''''''''''''''''''''''''''''''' Imports '''''''''''''''''''''''''''''''''''''''''''''''''"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import ticker, font_manager
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

"''''''''''''''''''''''''''''''''''''''''''''' Settings '''''''''''''''''''''''''''''''''''''''''''''''''"

xtick_font_size = 30
ytick_font_size = 30
xlabel_font_size = 40

xtick_font_size_3d = 15
ytick_font_size_3d = 15
xlabel_font_size_3d = 20

sub_title_fontsize = 25
suptitle_font_size = 45
title_font_size = 48
legend_font_size = 20

xtick_Padding = 15
ytick_Padding = 15
xlable_Padding = 80
ylable_Padding = 140
suptitle_Padding = 80
title_Padding = 30

"''''''''''''''''''''''''''''''''''''''''''''' Functions '''''''''''''''''''''''''''''''''''''''''''''''''"

def get_spike_count(raster_time, neuron_index):
    
    steps = list(np.arange(0,500*200,500))
    
    rvalues, _, ـ = np.unique(raster_time, True, return_counts=True)
    a = []
    for s in steps:
        f = np.where(np.logical_and(rvalues>=s, rvalues<=s+350))[0]
        
        a.append(len(np.unique(neuron_index[f])))

    return a


class Dec2ScalarFormatter(ticker.ScalarFormatter):
    def _set_format(self):
        self.format = '%.2f'  # Show 2 decimals

class Dec1ScalarFormatter(ticker.ScalarFormatter):

    def _set_format(self):
        self.format = '%.1f'  # Show 1 decimals

class intScalarFormatter(ticker.ScalarFormatter):

    def _set_format(self):
        self.format = '%.0f'  


def plot_raster(raster_time0, raster_neuron_index0,
                raster_time2, raster_neuron_index2,
                dataset, path=''):
    
    fig, axs = plt.subplots(2, sharex=True, figsize=(20, 7))
    
    p1 = axs[0].scatter(raster_time0, raster_neuron_index0, color=pipe0_color, s=10, label='Pipeline 0')
    p3 = axs[1].scatter(raster_time2, raster_neuron_index2, color=pipe12_color, s=10, label='Pipeline 1 and 2')
    
    axs[0].tick_params(axis='both', which='major', labelsize=xtick_font_size)
    axs[1].tick_params(axis='both', which='major', labelsize=xtick_font_size)
    
    axs[0].set_yticks(np.arange(0, max(raster_neuron_index0)+50, 400))
    axs[1].set_yticks(np.arange(0, max(raster_neuron_index0)+50, 400))
    
    
    if 'letter' not in dataset.lower():
        axs[1].set_xticks(np.arange(0, max(raster_time0)+1000, 20000))
    
            
    formatter = Dec2ScalarFormatter(useMathText=True)
    axs[1].xaxis.set_major_formatter(formatter)
    
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    formatter.set_scientific(True)
    axs[1].xaxis.offsetText.set_fontsize(24)
    
    
    plt.axis('on')
    
    figl, axl = plt.subplots(figsize=(8, 1))
    axl.axis(False)
    axl.legend([p1, p3],["Pipeline 0", "Pipelines 1 and 2"],
                prop={'size': legend_font_size}, markerscale=6, title_fontsize=legend_font_size, ncols=2)
    
    figl.savefig(path, dpi=1000)
    
    plt.show()

def plot_activity(raster_time0, raster_neuron_index0,
                raster_time2, raster_neuron_index2, dataset, path=''):

    a0 = get_spike_count(raster_time0, raster_neuron_index0)
    a2 = get_spike_count(raster_time2, raster_neuron_index2)
       
    fig, ax = plt.subplots(1, sharex=True, figsize=(20, 6), dpi = 200)
    p1 = ax.plot(a0, label='Pipeline 0', c=pipe0_color, lw=2)
    p3 = ax.plot(a2, label='Pipeline 1 and 2', c=pipe12_color, lw=2, alpha=0.8)
    
    
    ax.tick_params(axis='both', which='major', labelsize=xtick_font_size)
    
    if 'digit' in dataset.lower():
        rng = [0, 10, 20, 30, 40, 50]
    else:
        rng = [0, 10, 20, 30, 40, 50, 60]
        
    ax.set_yticks(rng)
    
    plt.savefig(path, bbox_inches='tight', dpi = 200, transparent=False)
    
    plt.show()

def plot_labeling(data, path='', rng=[0, 20, 40, 60, 80], change_y_tick_font=False):
    
    fig, axes = plt.subplots(figsize=(10,3), ncols=2, sharey=True)
    fig.tight_layout()
    index = data['Labels']
    column0 = data['labeling']
    column1 = data['relabeling']
    axes[0].barh(index, column0, align='center', color='#D1A7A0', zorder=0)
    axes[0].set_title('Labeling', fontsize=24, pad=15, color='#D1A7A0')
    axes[1].barh(index, column1, align='center', color='#899D78', zorder=0)
    axes[1].set_title('Update labeling', fontsize=24, pad=15, color='#899D78')

    axes[0].invert_xaxis() 

    # To show data from highest to lowest
    plt.gca().invert_yaxis()

    axes[0].set(yticks=data['Labels'], yticklabels=data['Labels'])
    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='y', colors='white') # tick color
    axes[1].tick_params(axis='y', colors='#D1A7A0') # tick color
    
    axes[0].set_xticks(rng)
    axes[0].set_xticklabels(rng)

    axes[1].set_xticks(rng)
    axes[1].set_xticklabels(rng)

    
    axes[0].bar_label(axes[0].containers[0], label_type='center', color='#fff', fontsize=10)
    axes[1].bar_label(axes[1].containers[0], label_type='center', color='#fff', fontsize=10)
    
    axes[0].grid(False)
    axes[1].grid(False)
    
    for ax in axes.flat:
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    axes[1].spines['left'].set_visible(True)  
    axes[1].spines['left'].set_color('white')
    
    if change_y_tick_font:
        yticks_font = font_manager.FontProperties(family='Tahoma')
        for tick in axes[0].get_yticklabels():
            tick.set_fontproperties(yticks_font)
        
    plt.yticks(fontname = "Times New Roman")
    for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
        label.set(fontsize=13, color='#000')
    for label in (axes[1].get_xticklabels()  ):
        label.set(fontsize=13, color='#000')
        
    
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
    plt.savefig(path, bbox_inches='tight', dpi = 200, transparent=False)
        
    plt.show()
    
def plot_cf_matrix(confusion_matrix, vmin, vmax, cmap, fontsize, path=''):
    
    fig, ax = plt.subplots(1,figsize=(6,5))
    sns.heatmap(confusion_matrix, annot=True,
                vmin=vmin, vmax=vmax, cmap=cmap,
                ax=ax, yticklabels=label_mapping,
                xticklabels=label_mapping,
                annot_kws={"fontsize":fontsize,'color':'#cecece'}) # font size 7 for letter

    ax.collections[0].colorbar.set_ticks([vmin, vmax])
    ax.tick_params(axis='y', rotation=0) 
    if 'letter' in dataset.lower():
        ticks_font = font_manager.FontProperties(family='Tahoma')
        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            xtick.set_fontproperties(ticks_font)
            ytick.set_fontproperties(ticks_font)
        
    plt.savefig(path, bbox_inches='tight', dpi = 200, transparent=False)
    
    plt.show()
    
def create_pd(data):
    
    # 400 neurons are cols and the time steps are indices
    df = pd.DataFrame(columns=list(range(0, 400)),
                      index=list(data.keys()))
    for t, v in  data.items():
       for n, ncs in v.items():
          df.at[t, n] = ncs
    return df.fillna(0)

def set_plt_params():
    plt.clf()
    matplotlib.rc_file_defaults()
    plt.rcParams['axes.facecolor'] = '#FFF'

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Myriad Pro']
    plt.rcParams["font.weight"] = "bold"
    
def plot_NCS_heatmap(pipe0_ncs, pipe1_ncs, pipe2_ncs, path='', dataset='digit'):
    
    df0 = create_pd(pipe0_ncs)

    df2 = create_pd(pipe2_ncs)

    sns.set(font_scale=5) 
    fig, axs = plt.subplots(2, 1, figsize=(40, 20),
                            sharey=True, sharex=True)
    
    for ax in axs:
        ax.set_facecolor('#cecece')
    cmap = 'afmhot'

    vmin = 0
    vmax = 0.00146
    vmax = np.max([np.max(x) for x in [df0, df2]])
    a1 = sns.heatmap(df0, yticklabels=15, xticklabels=50, label='Pipeline 0',
                     vmin=0, vmax=vmax, ax=axs[0], cmap=cmap, cbar=False)

    a3 = sns.heatmap(df2, yticklabels=15, xticklabels=50, label='Pipeline 2',
                     vmin=0, vmax=vmax, ax=axs[1], cmap=cmap, cbar=False)

    axs[1].invert_yaxis()
    

    axs[1].set_yticks([0, 25, 50])

    axs[1].set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])

    
    formatter = intScalarFormatter(useMathText=True)
    axs[1].yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    
    plt.savefig(r'{}\ncs_heatmap_{}.png'.format(path, dataset),
                bbox_inches='tight', dpi = 200, transparent=False)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(40, 20))
    mappable = a3.get_children()[0]
    fig.subplots_adjust(top=1.2)
    cax = fig.add_axes([0.022, 0.98, 0.965, 0.04])
    
    c_bar = plt.colorbar(mappable, cax=cax, ax=ax,
                         orientation='horizontal',
                         format=Dec1ScalarFormatter(useMathText=True))
    
    c_bar.formatter.set_powerlimits((0, 0))
    c_bar.formatter.set_useOffset(False)
    c_bar.formatter.set_scientific(True)


    ax.remove()
    plt.savefig(r'{}\ncs_heatmap_cbar_{}.png'.format(path, dataset),
                bbox_inches='tight', dpi = 200, transparent=False)
    
    plt.show()
    
def plot_acc(pipe0_df, pipe1_df, pipe2_df, dataset, path=''):
    
    if 'Noisy letter'.lower() in dataset.lower():
        k1s = [8.5]
        k2s = [10]
        ylim_lower = 30
        ylim_upper = 90
    
    elif 'Noisy digit'.lower() in dataset.lower():
        k1s = [9]
        k2s = [10.5]
        ylim_lower = 20
        ylim_upper = 80
    
    elif 'Digit'.lower() in dataset.lower():
        k1s = [10.5]
        k2s = [8]
        ylim_lower = 40
        ylim_upper = 90
    
    elif 'Letter'.lower() in dataset.lower():
        k1s = [10]
        k2s = [10]
        ylim_lower = 50
        ylim_upper = 90

    xs = range(1, 16, 1)

    fig, ax = plt.subplots(1, figsize=(14, 8))

    ax.plot(xs, pipe0_df['test acc'], label='Pipeline 0', 
            lw=10, linestyle=':', c=pipe0_color)


    gs = pipe1_df.groupby('k')
    i = 0
    for k, v in gs:
        if k in k1s:
            xmax = np.argmax(v['test acc'])
            ymax = np.max(v['test acc'])
            ax.plot(xs, v['test acc'], label='Pipeline 1'.format(k),
                    lw=10, linestyle=':', c=pipe1_color)


    gs = pipe2_df.groupby('K')
    for k, v in gs:
        if k in k2s:
            xmax = np.argmax(v['test acc'])
            ymax = np.max(v['test acc'])
            ax.plot(xs, v['test acc'], label='Pipeline 2',
                    lw=10, linestyle=':', c=pipe2_color)
            
    ax.set_ylabel('Accuracy', fontsize=xlabel_font_size)
    ax.set_xlabel('Epoch', fontsize=xlabel_font_size)

    ax.set_yticks(range(10, 91, 10))
    ax.set_xticks(xs)

    ax.set_ylim(ylim_lower, ylim_upper)

    ax.tick_params(axis='both', which='major', labelsize=xtick_font_size)
    
    ax.yaxis.grid(which="major", color='black', linestyle=':', linewidth=2, alpha=0.3)
    
    plt.savefig(r'{}\ptl_acc_{}_.png'.format(path, dataset),
                        bbox_inches='tight', dpi = 200, transparent=False)
    label_params = ax.get_legend_handles_labels() 

    figl, axl = plt.subplots(figsize=(6, 1))
    axl.axis(False)
    axl.legend(*label_params, ncols=4)
    figl.savefig(r'{}\legend_acc.png'.format(path), dpi=500)
    plt.show()
    
def plot_cf_matrix(confusion_matrix, vmin, vmax, label_mapping, cmap, fontsize, path=''):
    
    fig, ax = plt.subplots(1,figsize=(6,5))
    sns.heatmap(confusion_matrix, annot=True,
                vmin=vmin, vmax=vmax, cmap=cmap,
                ax=ax, yticklabels=label_mapping,
                xticklabels=label_mapping,
                annot_kws={"fontsize":fontsize,'color':'#cecece'}) # font size 7 for letter

    ax.collections[0].colorbar.set_ticks([vmin, vmax])
    ax.tick_params(axis='y', rotation=0) 
    if 'letter' in dataset.lower():
        ticks_font = font_manager.FontProperties(family='Tahoma')
        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            xtick.set_fontproperties(ticks_font)
            ytick.set_fontproperties(ticks_font)
        
    plt.savefig(path, bbox_inches='tight', dpi = 200, transparent=False)
    
    plt.show()
    
"''''''''''''''''''''''''''''''''''''''''''''' Main '''''''''''''''''''''''''''''''''''''''''''''''''"

main_dir = r'D:\Workspace\SNN\Revision codes and data\data\visualization data'
acc_data_dir = r'{}\\Accuracy_AllDataset'.format(main_dir)

pipe0_color = '#3EC300'
pipe1_color = '#4FCED7'
pipe2_color = '#DE2117'
pipe12_color = '#9c4fd7'

digit_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fashion_mapping = ['T-shirt/top'
,'Trouser'
,'Pullover'
,'Dress'
,'Coat'
,'Sandal'
,'Shirt'
,'Sneaker'
,'Bag'
,'Ankle boot']

letter_mapping = ['A'
,'B'
,'C'
,'D'
,'E'
,'F'
,'G'
,'H'
,'I'
,'J'
,'K'
,'L'
,'M'
,'N'
,'O'
,'P'
,'Q'
,'R'
,'S'
,'T'
,'U'
,'V'
,'W'
,'X'
,'Y'
,'Z'
,'a'
,'b'
,'d'
,'e'
,'f'
,'g'
,'h'
,'n'
,'q'
,'r'
,'t']


datasets =  ['digit', 'DigitNoisy',
             'letter', 'LetterNoisy'
             ]
# to read from the acc excel sheet correctly
acc_excel_name_mapping = {
     'LetterNoisy': 'Noisy letter',
     'DigitNoisy': 'Noisy digit',
     'digit': 'Digit',
     'letter': 'Letter',
    }
for dataset in tqdm(datasets):
        
    pipeline = 'pipe0'
    

    set_plt_params()
    
    # visualize based on input data (train\test)
    raster_neuron_index0 = np.load(r'{}\{}\{}\raster_neuronIndex_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    raster_time0 = np.load(r'{}\{}\{}\raster_time_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    input_numbers0 = np.load(r'{}\{}\{}\Original_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    predictions0 = np.load(r'{}\{}\{}\predicted_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    
    pipeline = 'pipe1'
    input_numbers1 = np.load(r'{}\{}\{}\Original_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    predictions1 = np.load(r'{}\{}\{}\predicted_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    
    pipeline = 'pipe2'
    
    raster_neuron_index2 = np.load(r'{}\{}\{}\raster_neuronIndex_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    raster_time2 = np.load(r'{}\{}\{}\raster_time_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    pipeline = 'pipe2'
    predictions2 = np.load(r'{}\{}\{}\predicted_{}_{}.npy'.format(
        main_dir, dataset, pipeline, pipeline, dataset))
    
    
    ############## raster and activity #################
    
    
    if 'Digit' in dataset:
        label_mapping = digit_mapping
    elif 'Letter' in dataset:
        label_mapping = letter_mapping
    else:
        label_mapping = fashion_mapping
    
    plot_raster(raster_time0, raster_neuron_index0,
                    raster_time2, raster_neuron_index2, dataset, path='')
    
    plot_activity(raster_time0, raster_neuron_index0,
                    raster_time2, raster_neuron_index2, dataset, path='')
    
    
    ############## CF matrix #################
    
    digit_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fashion_mapping = ['T-shirt/top'
    ,'Trouser'
    ,'Pullover'
    ,'Dress'
    ,'Coat'
    ,'Sandal'
    ,'Shirt'
    ,'Sneaker'
    ,'Bag'
    ,'Ankle boot']
    
    letter_mapping = ['E', 'I', 'l', 'w', 'q', 'Q', 'u', 's', 'b' , 'g']
    
    if 'digit' in dataset.lower():
        label_mapping = digit_mapping
    
    elif 'letter' in dataset.lower() or 'LetterNoisy' in dataset.lower():
        label_mapping = letter_mapping
    else:
        label_mapping = fashion_mapping
    
    confusion_matrix0 = metrics.confusion_matrix(input_numbers0, predictions0)
    confusion_matrix1 = metrics.confusion_matrix(input_numbers1, predictions1)
    confusion_matrix2 = metrics.confusion_matrix(input_numbers1, predictions2)
    
    cmap = sns.dark_palette("#18C1A4", as_cmap=True)
    sns.set(font_scale=1.5) # 0.8 for letter
    
    
    vmax = max([np.max(x) for x in [confusion_matrix0, confusion_matrix1, confusion_matrix2]])
    vmin = min([np.min(x) for x in [confusion_matrix0, confusion_matrix1, confusion_matrix2]])
        
    plot_cf_matrix(confusion_matrix0, vmin, vmax, label_mapping, cmap, 12, '')
    plot_cf_matrix(confusion_matrix1, vmin, vmax, label_mapping, cmap, 12, '')
    plot_cf_matrix(confusion_matrix2, vmin, vmax, label_mapping, cmap, 12, '')
    
    ############### Accuracy #################
    
    set_plt_params()
    
    pipe0_df = pd.ExcelFile(r'{}\Results_pipe0.xlsx'.format(acc_data_dir))
    pipe1_df = pd.ExcelFile(r'{}\Results_pipe1.xlsx'.format(acc_data_dir))
    pipe2_df = pd.ExcelFile(r'{}\Results_pipe2.xlsx'.format(acc_data_dir))
    train_df = pd.ExcelFile(r'{}\Results_training.xlsx'.format(acc_data_dir))
    
    dataset_sheet_name = acc_excel_name_mapping[dataset]
    
    dfs = {sheet_name: pipe0_df.parse(sheet_name) 
              for sheet_name in pipe0_df.sheet_names}
    pipe0_df = dfs[dataset_sheet_name]

    dfs = {sheet_name: pipe1_df.parse(sheet_name) 
              for sheet_name in pipe1_df.sheet_names}
    pipe1_df = dfs[dataset_sheet_name]

    dfs = {sheet_name: pipe2_df.parse(sheet_name) 
              for sheet_name in pipe2_df.sheet_names}
    pipe2_df = dfs[dataset_sheet_name]
    
    plot_acc(pipe0_df, pipe1_df, pipe2_df, dataset_sheet_name)
    
    ############## NCS #################
    
    
    p0_dir = r'{}\SAM\50_time_steps_final\ncs_{}_{}.pickle'.format(main_dir, 
                                                                 dataset,
                                                                 'pipe0')
 
        
    pipe0_ncs = pd.read_pickle(p0_dir)
    
    # pipe 1
    pipe1_ncs = pd.read_pickle(r'{}\SAM\50_time_steps_final\ncs_{}_{}.pickle'.format(main_dir, 
                                                                 dataset,
                                                                 'pipe1'))
    # pipe 2
    pipe2_ncs = pd.read_pickle(r'{}\SAM\50_time_steps_final\ncs_{}_{}.pickle'.format(main_dir, 
                                                                 dataset,
                                                                 'pipe2'))
    
    plot_NCS_heatmap(pipe0_ncs, pipe1_ncs, pipe2_ncs, '', dataset)
    
   
    
############### labeling #################

set_plt_params() 

Labeling_Emnist_10 = [64, 45, 23, 36, 34, 48, 45, 47, 33, 25]
Relabeling_Emnist_10 = [45, 38, 21, 44, 38, 47, 44, 48, 38, 37]
letter10_mapping = ['E', 'I', 'l', 'w', 'q', 'Q', 'u', 's', 'b', 'g']

Labeling_digit = [62, 28, 40, 35, 35, 44, 34, 37, 44, 41]
Relabeling_digit = [50, 29, 46, 44, 39, 35, 43, 38, 45, 31]
digit_mapping = [str(x) for x in range(0,10)]


emnist_pd = pd.DataFrame(list(zip(letter10_mapping,
                                  Labeling_Emnist_10,
                                  Relabeling_Emnist_10)),
                         columns=['Labels', 'labeling', 'relabeling'])

digit_pd = pd.DataFrame(list(zip(digit_mapping,
                                  Labeling_digit,
                                  Relabeling_digit)),
                         columns=['Labels', 'labeling', 'relabeling'])

plot_labeling(emnist_pd, r'', change_y_tick_font=True, rng=[0, 20, 40, 65])
plot_labeling(digit_pd, r'', rng=[0, 20, 40, 65])