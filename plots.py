import re
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import os
import json
import wandb

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
    
def Rearrange_PAT_csv_files(dir_path:str = 'saved_models'):
    """
    Rearrange the csv files in the given directory to be in a more organized way.
    """
    matches = []
    for parent, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.csv'):
            if 'alexnet' in parent or 'self' in parent:
                matches.append(os.path.join(parent, filename))
            else:
                continue
    for file in matches:
        # Load the csv using pandas to a dataframe
        df = pd.read_csv(file)
        # Create a new dataframe, where the first column named `Epsilon`, and it consist of the values 0 to 64, included.
        # The second columns should include the values from the first row of the above `df`.
        new_df = pd.DataFrame({'epsilon': np.arange(1, 65, 1), 'eval_results': df.iloc[0].values})
        # save the new_df in the same path as file under the name `eval_accuracy_8.csv`
        file_dirs = file.split('/')
        save_path = '/'.join(file_dirs[:-1]) + '/eval_accuracy_8.csv'
        new_df.to_csv(save_path, index=False)

def Plotly_plot_data(csv_files:list, y_label_column:str, specific_epsilons:list[int] = None, specific_method:str=None, specific_agnostic=None):
    """
    Plot a figure for a specific set of tested model, setting, training epsilon and loss function.
    Plot the figure a function of `y_label_column` w.r.t evaluated epsilons.
    """
    fig = go.Figure()
    for idx, file in enumerate(csv_files):
        if file.endswith('evaluation.csv'):
            continue
        df = pd.read_csv(file)
        epsilons = df['epsilon']
        column_results = df[y_label_column]
        if 'results' in y_label_column:
            column_results = column_results * 100
        if 'alexnet' in file or 'self' in file:
            if 'alexnet' in file:
                label = 'Pat-AlexNet-0.7'
            elif 'self' in file:
                label = 'Pat-Self-0.5'
            else:
                raise ValueError('Model name not found')
        else:
            file_dirs = file.split('/')
            model_name = file_dirs[1]
            pgd_steps_dir = [dir for dir in file_dirs if 'pgd_steps_' in dir]
            if len(pgd_steps_dir) == 0:
                pgd_steps = 10
            else:
                pgd_steps = int(pgd_steps_dir[0].split('pgd_steps_')[-1])
            seed = file_dirs[2] if 'sanity_check' not in file else file_dirs[3]
            eval_trained_epsilon = re.search(r'[\d][\d]?', file_dirs[-1]).group()
            target_agnostic_dir = [dir for dir in file_dirs if 'agnostic_loss_' in dir][0]
            whether_agnostic = 'True' if 'True' in target_agnostic_dir else 'False'
            res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
            train_method = None
            if 'adaptive' in file:
                train_method = 'adaptive'
            elif 're_introduce' in file:
                train_method = 're_introduce'
            elif 'train' in file:
                train_method = 'vanilla'
            # train_method = train_method if train_method != 'train' else 'Vanilla'
            if specific_epsilons is not None and int(eval_trained_epsilon) not in specific_epsilons:
                continue
            if specific_method is not None and train_method != specific_method:
                continue
            if specific_agnostic is not None and whether_agnostic != specific_agnostic:
                continue
            label = f'Model: {model_name}, '
            label += f'PGD steps: {pgd_steps}, '
            label += f'Method: {train_method}, ' if specific_method is None else ''
            label += f'Agnostic: {whether_agnostic}, ' if specific_agnostic is None else ''
            label += f'Train_epsilon: {eval_trained_epsilon}'
            label += f' - old' if 'old' in file else ''
            if 'sanity_check_2' in file:
                label = 'Sanity Check 2 - ' + label
            elif 'sanity_check' in file:
                label = 'Sanity Check - ' + label
        fig.add_trace(go.Scatter(x=epsilons, y=column_results, mode='lines', name=label))

    fig.update_layout(
        xaxis_title='Epsilons',
        yaxis_title='Test Accuracy' if y_label_column == 'eval_results' else 'Train Accuracy' if y_label_column == 'train_results' else y_label_column,
        title=f'{titles_names[y_label_column]} as a function of Epsilons' + (
            f' for models trained on Epsilons: {str(specific_epsilons)[1:-1]}' if specific_epsilons is not None else ''
        ) + (
            f' with {specific_method} method' if specific_method is not None else ''
        ) + (
            f' with agnostic={specific_agnostic}' if specific_agnostic is not None else ''
        ),
        xaxis=dict(tickmode='linear', tick0=0, dtick=4),
        yaxis=dict(range=[0, 100], dtick=10),
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5,
                bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='Black', borderwidth=1),
        margin=dict(l=40, r=40, t=40, b=100)
    )

    filename_suffix = f'_trained_on_{str(specific_epsilons)[1:-1]}' if specific_epsilons is not None else ''
    filename_suffix += f'_with_{specific_method}_method' if specific_method is not None else ''
    filename_suffix += f'_with_agnostic={specific_agnostic}' if specific_agnostic is not None else ''
    # if not os.path.exists(f'pdf_plots/{y_label_column}'):
    #     os.makedirs(f'pdf_plots/{y_label_column}')
    # fig.write_image(f'pdf_plots/{y_label_column}/models{filename_suffix}.pdf')
    if not os.path.exists(f'html_plots/{y_label_column}'):
        os.makedirs(f'html_plots/{y_label_column}')
    fig.write_html(f'html_plots/{y_label_column}/models{filename_suffix}.html')
    # fig.show()


def Wandb_report_data(csv_files:list):
    """
    Given a list of csv files, report the data to wandb.
    Each csv should result in a different run.
    Report every result column in the csv with respect to the epsilon values.
    """
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(file)
        epsilons = df['epsilon']
        file_dirs = file.split('/')
        model_name = file_dirs[1]
        pgd_steps_dir = [dir for dir in file_dirs if 'pgd_steps_' in dir]
        if len(pgd_steps_dir) == 0:
            pgd_steps = 10
        else:
            pgd_steps = int(pgd_steps_dir[0].split('pgd_steps_')[-1])
        seed = file_dirs[2] if 'sanity_check' not in file else file_dirs[3]
        eval_trained_epsilon = re.search(r'[\d][\d]?', file_dirs[-1]).group()
        target_agnostic_dir = [dir for dir in file_dirs if 'agnostic_loss_' in dir][0]
        whether_agnostic = 'True' if 'True' in target_agnostic_dir else 'False'
        res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
        train_method = None
        if 'adaptive' in file:
            train_method = 'adaptive'
        elif 're_introduce' in file:
            train_method = 're_introduce'
        elif 'train' in file:
            train_method = 'vanilla'
        args = {
            'model_name': model_name,
            'pgd_steps': pgd_steps,
            'seed': seed,
            'eval_trained_epsilon': eval_trained_epsilon,
            'whether_agnostic': whether_agnostic,
            'train_method': train_method,
            'Inference': True
        }
        args['log_name'] = f"{args['model_name']}_train method_{args['train_method']}_agnostic_loss_{args['whether_agnostic']}_seed_{args['seed']}_max epsilon_{int(args['eval_trained_epsilon'])}"
        run = wandb.init(project="Adversarial-adaptive-project", name=f"Inference_{args['log_name']}", entity = "ido-shani-proj", config=args)
        # Iterate over the values but from row number 1, since row number zero is columns titles:
        for idx, (epsilon, eval, train) in enumerate(zip(df['epsilon'], df['eval_results'], df['train_results'])):
            run.log({'Inference/ Test': eval*100}, step=epsilon)
            run.log({'Inference/ Train': train*100}, step=epsilon)
        run.finish()

def plot_data(csv_files:list, y_label_column:str, specific_epsilons:list[int] = None, specific_method:str=None, specific_agnostic=None): 
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
        model_name = file_dirs[1]
        # eval_trained_epsilon = args.eval_model_path.split('/')[-1]
        eval_trained_epsilon = re.search('[\d][\d]?', file_dirs[-1]).group()
        whether_agnostic = 'True' if 'True' in file_dirs[-2] else 'False'
        res = [i for i in range(len(file_dirs[-3])) if file_dirs[-3].startswith('_', i)]
        train_method = file_dirs[-3][res[1]+1:]
        train_method =  train_method if train_method!='train' else 'Vanilla'
        if specific_epsilons is not None and int(eval_trained_epsilon) not in specific_epsilons:
            continue
        if specific_method is not None and train_method != specific_method:
            continue
        if specific_agnostic is not None and whether_agnostic != specific_agnostic:
            continue
        label = f'Model: {model_name}, '
        label += f'Method: {train_method}, ' if specific_method is None else ''
        label += f'Agnostic: {whether_agnostic}, ' if specific_agnostic is None else ''
        # label += f'Train_epsilon: {eval_trained_epsilon}' if specific_epsilon is None else ''
        label += f'Train_epsilon: {eval_trained_epsilon}'
        if 'sanity_check_2' in file:
            label = 'Sanity Check 2 - ' + label
        elif 'sanity_check' in file:
            label = 'Sanity Check - ' + label
        plt.plot(epsilons, column_results, label=label)
    plt.xlabel('Epsilons')
    if y_label_column=='eval_results':
        plt.ylabel('Test Accuracy')
    elif y_label_column=='train_results':
        plt.ylabel('Train Accuracy')
    else:
        plt.ylabel(y_label_column)
    # Use tick marks to show the epsilon values
    xticks_arg = max(specific_epsilons)+1 if specific_epsilons is not None else 65
    plt.xticks(np.arange(0, xticks_arg, 4))
    title_suffix = f' for models trained on Epsilons: {str(specific_epsilons)[1:-1]}' if specific_epsilons is not None else ''
    title_suffix += f' with {specific_method} method' if specific_method is not None else ''
    title_suffix += f' with agnostic={specific_agnostic}' if specific_agnostic is not None else ''
    plt.title(f'{titles_names[y_label_column]} as a function of Epsilons'+title_suffix)
    plt.legend()
    plt.grid()
    filename_suffix = f'_trained_on_{str(specific_epsilons)[1:-1]}' if specific_epsilons is not None else ''
    filename_suffix += f'_with_{specific_method}_method' if specific_method is not None else ''
    filename_suffix += f'_with_agnostic={specific_agnostic}' if specific_agnostic is not None else ''
    if not os.path.exists(f'pdf_plots/{y_label_column}'):
        os.makedirs(f'pdf_plots/{y_label_column}')
    plt.savefig(f'pdf_plots/{y_label_column}/models{filename_suffix}.pdf')
    plt.show()
    # Close the plt figure
    plt.close()

if __name__ == '__main__':
    # # Used the below to plot the left figure in Figure 2 
    # files_list = ['saved_models/resnet18/seed_42/train_method_adaptive/agnostic_loss_True/eval_accuracy_32.csv',
    #               'saved_models/resnet18/seed_42/train_method_re_introduce/agnostic_loss_True/eval_accuracy_16.csv',
    #               'saved_models/resnet18/seed_42/train_method_train/agnostic_loss_False/eval_accuracy_32.csv']
    # plot_specific_data(files_list, 'eval_results', 'best_results')
    
    # # Find all csv files in current folder and its subfolder to plot every possible figure.
    dir_path = 'saved_models'
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
    # for y_label_column in ['train_results', 'eval_results', 'AUROC', 'AURC', 'ece15', 'confidence_mean']:
    #     plot_data(matches, y_label_column)
    #     plot_data(matches, y_label_column, specific_epsilon='8')
    #     plot_data(matches, y_label_column, specific_epsilon='16')
    #     plot_data(matches, y_label_column, specific_epsilon='32')
    #     plot_data(matches, y_label_column, specific_method='adaptive')
    #     plot_data(matches, y_label_column, specific_method='re_introduce')
    #     plot_data(matches, y_label_column, specific_agnostic='True')
    #     plot_data(matches, y_label_column, specific_agnostic='False')

    # Testing sanity case
    # matches = ['saved_models/resnet18/sanity_check/seed_42/train_method_re_introduce/agnostic_loss_False/eval_accuracy_32.csv']
    # plot_data(matches, 'eval_results', specific_method='adaptive', specific_epsilons=[32, 64])
    matches = [match for match in matches if 'WideResNet' not in match and ('accuracy_32' in match or 'accuracy_64' in match)]
    # # Plotly_plot_data(matches, 'eval_results', specific_method='adaptive', specific_epsilons=[32, 64])
    # Plotly_plot_data(matches, 'eval_results', specific_epsilons=[32, 64])
    # Rearrange_PAT_csv_files()
    Wandb_report_data(matches)