import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import seaborn as sns

import fnmatch


def plot_data(matches, y_label_column, specific_epsilon = None, specific_method=None, specific_agnostic=None): 
    
    # Each csv contains epsilons, train and test accuracies.
    # Create a plot for train accuracy and test accuracy as a function of epsilons.
    # The train accuracies from all csv files should be in the same plot, and all the test accuracies should be in another single plot.
    # The plot should have a legend, title and labels.
    # Save the plot as a png file.
    # print(matches)
    # exit()

    for idx,file in enumerate(matches):
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
    if not os.path.exists(f'plots/{y_label_column}'):
        os.makedirs(f'plots/{y_label_column}')
    plt.savefig(f'plots/{y_label_column}/models{filename_suffix}.png')
    plt.show()
    # Close the plt figure
    plt.close()

    # for idx,file in enumerate(matches):
    #     df = pd.read_csv(file)
    #     epsilons = df['epsilon']
    #     eval_results = df['eval_results']*100
    #     file_dirs = file.split('/')
    #     eval_trained_epsilon = re.search('[\d][\d]?', file_dirs[-1]).group()
    #     whether_agnostic = 'True' if 'True' in file_dirs[-2] else 'False'
    #     res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
    #     train_method = file_dirs[-3][res[1]+1:]
    #     train_method =  train_method if train_method!='train' else 'Vanilla'
    #     if specific_epsilon is not None and eval_trained_epsilon != specific_epsilon:
    #         continue
    #     if specific_method is not None and train_method != specific_method:
    #         continue
    #     if specific_agnostic is not None and whether_agnostic != specific_agnostic:
    #         continue
    #     label = '' + f'Method: {train_method}, ' if specific_method is None else ''
    #     label += f'Agnostic: {whether_agnostic}, ' if specific_agnostic is None else ''
    #     label += f'Train_epsilon: {eval_trained_epsilon}' if specific_epsilon is None else ''
    #     plt.plot(epsilons, eval_results, label=label)
    # plt.xlabel('Epsilons')
    # plt.ylabel('Accuracy')
    # plt.xticks(np.arange(0, 33, 2))
    
    # plt.title('Test Accuracy as a function of Epsilons'+title_suffix)
    # plt.legend()
    # plt.grid()
    # plt.savefig(f'plots/Test_results{filename_suffix}.png')
    # plt.close()


    # Uncertainty plots
    """
    plt.plot(coverages[selected_indices_to_present].tolist(),
                 selective_risks[selected_indices_to_present].tolist(), 
                # best_selective_risks[selected_indices_to_present].tolist(),
                label='rc_curve')
        plt.xlabel('Coverage')
        plt.ylabel('Selective Risk')
        
        plt.title('Selective risk as a function of coverage')
        plt.legend()
        plt.grid()
        save_dir = 'plots/uncertainty'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f'{save_dir}/test.png')
        plt.close()
    """


if __name__ == '__main__':
    # Find all csv files in current folder and its subfolder
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