import re
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import fnmatch


def plot_specific_data(files_list, y_label_column, fig_save_name):
    """
    This function was created only to plot the left figure in Figure 2, to look as similar as possible to the one in CAT.
    Used for comparison between the our trained models and the models in CAT.
    """
    for idx,file in enumerate(files_list):
        df = pd.read_csv(file)
        epsilons = df['epsilon']
        epsilons = epsilons/255*100
        column_results = df[y_label_column]
        if 'results' in y_label_column:
            column_results = column_results*100
        file_dirs = file.split('/')
        eval_trained_epsilon = re.search('[\d][\d]?', file_dirs[-1]).group()
        whether_agnostic = 'True' if 'True' in file_dirs[-2] else 'False'
        res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
        train_method = file_dirs[-3][res[1]+1:]
        train_method =  train_method if train_method!='train' else 'Vanilla'
        label = '' + f'Method: {train_method}, '
        label += f'Agnostic: {whether_agnostic}, '
        label += f'Train_epsilon: {eval_trained_epsilon}'
        plt.plot(epsilons, column_results, label=label)
    plt.ylim(-5, 101)
    plt.xlabel('Epsilons', fontsize=14)
    plt.yticks(fontsize=12)
    if y_label_column=='eval_results':
        plt.ylabel('Test Accuracy', fontsize=14)
    elif y_label_column=='train_results':
        plt.ylabel('Train Accuracy', fontsize=14)
    else:
        plt.ylabel(y_label_column, fontsize=14)
    # Use tick marks to show the epsilon values
    # Create an instance of AnchoredText to add a text box with the epsilon values magnitude on the X-axis.
    ax = plt.gca()
    anchored_text = AnchoredText("(1e-2)", loc='lower right', frameon=False, bbox_to_anchor=(1.08, -0.07),
                                bbox_transform=ax.transAxes, pad=0.1, prop=dict(color='black', fontsize=10))
    plt.gca().add_artist(anchored_text)

    plt.xticks(np.arange(0, 13, 1))
    plt.title(f'Accuracy as a function of Epsilons - ResNet-18 PGD10')
    plt.legend()
    plt.grid()

    if not os.path.exists(f'pdf_plots'):
        os.makedirs(f'pdf_plots')
    plt.savefig(f'pdf_plots/{fig_save_name}.pdf')
    plt.show()
    # Close the plt figure
    plt.close()
    


def plot_data(csv_files:list, y_label_column:str, specific_epsilon = None, specific_method:str=None, specific_agnostic=None): 
    """
    Plot a figure for a specific set of tested model, setting, training epsilon and loss function.
    Plot the figure a function of `y_label_column` w.r.t evaluated epsilons.
    """
    for idx,file in enumerate(csv_files):
        df = pd.read_csv(file)
        epsilons = df['epsilon']
        column_results = df[y_label_column]
        if 'results' in y_label_column:
            column_results = column_results*100
        file_dirs = file.split('/')
        # eval_trained_epsilon = args.eval_model_path.split('/')[-1]
        eval_trained_epsilon = re.search('[\d][\d]?', file_dirs[-1]).group()
        whether_agnostic = 'True' if 'True' in file_dirs[-2] else 'False'
        res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
        train_method = file_dirs[-3][res[1]+1:]
        train_method =  train_method if train_method!='train' else 'Vanilla'
        if specific_epsilon is not None and eval_trained_epsilon != specific_epsilon:
            continue
        if specific_method is not None and train_method != specific_method:
            continue
        if specific_agnostic is not None and whether_agnostic != specific_agnostic:
            continue
        label = '' + f'Method: {train_method}, ' if specific_method is None else ''
        label += f'Agnostic: {whether_agnostic}, ' if specific_agnostic is None else ''
        label += f'Train_epsilon: {eval_trained_epsilon}' if specific_epsilon is None else ''
        plt.plot(epsilons, column_results, label=label)
    plt.xlabel('Epsilons')
    if y_label_column=='eval_results':
        plt.ylabel('Test Accuracy')
    elif y_label_column=='train_results':
        plt.ylabel('Train Accuracy')
    else:
        plt.ylabel(y_label_column)
    # Use tick marks to show the epsilon values
    plt.xticks(np.arange(0, 33, 2))
    title_suffix = f' for models trained on Epsilon: {specific_epsilon}' if specific_epsilon is not None else ''
    title_suffix += f' with {specific_method} method' if specific_method is not None else ''
    title_suffix += f' with agnostic={specific_agnostic}' if specific_agnostic is not None else ''
    plt.title(f'{titles_names[y_label_column]} as a function of Epsilons'+title_suffix)
    plt.legend()
    plt.grid()
    filename_suffix = f'_trained_on_{specific_epsilon}' if specific_epsilon is not None else ''
    filename_suffix += f'_with_{specific_method}_method' if specific_method is not None else ''
    filename_suffix += f'_with_agnostic={specific_agnostic}' if specific_agnostic is not None else ''
    if not os.path.exists(f'pdf_plots/{y_label_column}'):
        os.makedirs(f'pdf_plots/{y_label_column}')
    plt.savefig(f'pdf_plots/{y_label_column}/models{filename_suffix}.pdf')
    plt.show()
    # Close the plt figure
    plt.close()

if __name__ == '__main__':
    # Used the below to plot the left figure in Figure 2 
    files_list = ['saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_True/eval_accuracy_32.csv',
                  'saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/eval_accuracy_16.csv',
                  'saved_models/resnet18/seed_42/train_method_train/agnostic_loss_False/eval_accuracy_32.csv']
    plot_specific_data(files_list, 'eval_results', 'best_results')
    
    # Find all csv files in current folder and its subfolder to plot every possible figure.
    dir_path = '.'
    matches = []
    for parent, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.csv'):
            matches.append(os.path.join(parent, filename))
    titles_names = {
        'train_results': 'Train Accuracy',
        'eval_results': 'Test Accuracy',
        'AUROC': 'AUROC',
        'AURC': 'AURC',
        'ece15': 'ECE15',
        'confidence_mean': 'Confidence',
    }
    for y_label_column in ['train_results', 'eval_results', 'AUROC', 'AURC', 'ece15', 'confidence_mean']:
        plot_data(matches, y_label_column)
        plot_data(matches, y_label_column, specific_epsilon='8')
        plot_data(matches, y_label_column, specific_epsilon='16')
        plot_data(matches, y_label_column, specific_epsilon='32')
        plot_data(matches, y_label_column, specific_method='adaptive')
        plot_data(matches, y_label_column, specific_method='re_introduce')
        plot_data(matches, y_label_column, specific_agnostic='True')
        plot_data(matches, y_label_column, specific_agnostic='False')