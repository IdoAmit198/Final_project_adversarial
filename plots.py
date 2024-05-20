import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import fnmatch


def plot_data(dir_path):
    # Find all csv files in current folder and its subfolder
    matches = []
    for parent, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.csv'):
            matches.append(os.path.join(parent, filename))
    # Each csv contains epsilons, train and test accuracies.
    # Create a plot for train accuracy and test accuracy as a function of epsilons.
    # The train accuracies from all csv files should be in the same plot, and all the test accuracies should be in another single plot.
    # The plot should have a legend, title and labels.
    # Save the plot as a png file.
    for idx,file in enumerate(matches):
        df = pd.read_csv(file)
        epsilons = df['epsilon']
        train_results = df['train_results']*100
        file_dirs = file.split('/')
        # eval_trained_epsilon = args.eval_model_path.split('/')[-1]
        eval_trained_epsilon = re.search('[\d][\d]?', file_dirs[-1]).group()
        whether_agnostic = 'True' if 'True' in file_dirs[-2] else 'False'
        res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
        train_method = file_dirs[-3][res[1]+1:]
        train_method =  train_method if train_method!='train' else 'Vanilla'
        plt.plot(epsilons, train_results, label=f'Method: {train_method}, Agnostic: {whether_agnostic}, Train_epsilon: {eval_trained_epsilon}')
    plt.xlabel('Epsilons')
    plt.ylabel('Accuracy')
    # Use tick marks to show the epsilon values
    plt.xticks(np.arange(0, 21, 2))
    plt.title('Train Accuracy as a function of Epsilons')
    plt.legend()
    plt.grid()
    plt.savefig(f'Train.png')
    plt.show()
    # Close the plt figure
    plt.close()

    for idx,file in enumerate(matches):
        df = pd.read_csv(file)
        epsilons = df['epsilon']
        eval_results = df['eval_results']*100
        file_dirs = file.split('/')
        eval_trained_epsilon = re.search('[\d][\d]?', file_dirs[-1]).group()
        whether_agnostic = 'True' if 'True' in file_dirs[-2] else 'False'
        res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
        train_method = file_dirs[-3][res[1]+1:]
        train_method =  train_method if train_method!='train' else 'Vanilla'
        plt.plot(epsilons, eval_results, label=f'Method: {train_method}, Agnostic: {whether_agnostic}, Train_epsilon: {eval_trained_epsilon}')
    plt.xlabel('Epsilons')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, 21, 2))
    plt.title('Test Accuracy as a function of Epsilons')
    plt.legend()
    plt.grid()
    plt.savefig(f'Test.png')
    plt.show()





if __name__ == '__main__':
    plot_data('.')