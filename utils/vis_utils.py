import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def draw_acc_hot(num_tasks,init_cls,increment,acc_table,save_path):
    title = 'Results on each task'
    xlabel = generate_labels(num_tasks, init_cls, increment)
    
    data_array = np.array(acc_table)

    plt.figure(figsize=(10, 8))
    sns.heatmap(data_array, cmap='Blues',mask=data_array==0,cbar=False, 
                annot=True, fmt=".1f", linewidths=.8, square=True, xticklabels=xlabel, annot_kws={'size':16})
    plt.set_cmap('Blues_r')
    plt.gca().patch.set_facecolor('lightgrey')
    plt.tick_params(axis='x', labelrotation=45)

    plt.title(title, fontsize=24)
    plt.xlabel('Classes seen so far', fontsize=24)
    plt.ylabel('Incremental Stage', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'vis', 'hot.png'))
    
def draw_acc_line(dataset,engine,num_tasks,acc_table,save_path):
    title = f'FAA on {dataset}'
    tasks = ['task{}'.format(i) for i in range(num_tasks)]
    data = acc_table[-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(tasks, data, marker='o', label=engine)

    plt.title(title,fontsize=24)
    plt.xlabel('Incremental Stage',fontsize=24)
    plt.ylabel('Final Average Accuracy(%)',fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'vis', 'line.png'))
    
def generate_labels(num_tasks, init_cls, increment):
    labels = []
    start = 0

    for i in range(num_tasks):
        add = init_cls if i == 0 else increment
        end = start + add - 1
        labels.append(f"{start}-{end}")
        start = end + 1
    
    return labels