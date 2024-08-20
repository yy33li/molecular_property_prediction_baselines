# import modules
import numpy as np
import pandas as pd
from scipy.stats import ranksums
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

#load the evaluation metrics results of all three frameworks on each dataset
bbbp = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "roc1": [0.898, 0.886, 0.764],
    "roc2": [0.945, 0.852, 0.764],
    "roc3": [0.914, 0.809, 0.772],
    "pr1": [0.953, 0.951, 0.852],
    "pr2": [0.982, 0.936, 0.852],
    "pr3": [0.976, 0.933, 0.846]
}

bace = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "roc1": [0.827, 0.655, 0.778],
    "roc2": [0.847, 0.655, 0.754],
    "roc3": [0.835, 0.655, 0.749],
    "pr1": [0.862, 0.771, 0.864],
    "pr2": [0.874, 0.771, 0.891],
    "pr3": [0.873, 0.771, 0.852]
}

clintox = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "roc1": [0.716, 0.876, 0.835],
    "roc2": [0.773, 0.852, 0.829],
    "roc3": [0.702, 0.674, 0.888],
    "pr1": [0.668, 0.790, 0.599],
    "pr2": [0.591, 0.602, 0.622],
    "pr3": [0.623, 0.543, 0.689]
}

sider = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "roc1": [0.693, 0.683, 0.710],
    "roc2": [0.669, 0.631, 0.713],
    "roc3": [0.672, 0.647, 0.685],
    "pr1": [0.699, 0.688, 0.746],
    "pr2": [0.656, 0.632, 0.718],
    "pr3": [0.708, 0.677, 0.703]
}

tox21 = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "roc1": [0.831, 0.831, 0.601],
    "roc2": [0.836, 0.825, 0.927],
    "roc3": [0.834, 0.822, 0.698],
    "pr1": [0.446, 0.407, 0.208],
    "pr2": [0.462, 0.401, 0.372],
    "pr3": [0.426, 0.367, 0.134]
}

esol = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "rmse1": [0.673, 1.263, 1.069],
    "rmse2": [0.651, 1.394, 1.059],
    "rmse3": [0.673, 1.118, 0.91],
    "correlation1": [0.946, 0.818, 0.950],
    "correlation2": [0.952, 0.781, 0.893],
    "correlation3": [0.939, 0.866, 0.929]
}

lipo = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "rmse1": [0.667, 0.897, 0.650],
    "rmse2": [0.595, 0.710, 0.616],
    "rmse3": [0.597, 0.649, 0.594],
    "correlation1": [0.831, 0.846, 0.846],
    "correlation2": [0.864, 0.825, 0.856],
    "correlation3": [0.873, 0.847, 0.871]
}

freesolv = {
    "models": ["FPGNN", "HiGNN", "TransFoxMol"],
    "rmse1": [1.407, 2.03, 1.231],
    "rmse2": [0.948, 1.575, 1.342],
    "rmse3": [1.135, 1.369, 1.188],
    "correlation1": [0.92, 0.824, 0.932],
    "correlation2": [0.967, 0.903, 0.935],
    "correlation3": [0.955, 0.937, 0.951]
}

#convert data to DataFrames that is easier for calculation
bbbp_df = pd.DataFrame(bbbp)
bace_df = pd.DataFrame(bace)
clintox_df = pd.DataFrame(clintox)
sider_df = pd.DataFrame(sider)
tox21_df = pd.DataFrame(tox21)
esol_df = pd.DataFrame(esol)
lipo_df = pd.DataFrame(lipo)
freesolv_df = pd.DataFrame(freesolv)

bbbp_df.set_index("models", inplace=True)
bace_df.set_index("models", inplace=True)
clintox_df.set_index("models", inplace=True)
sider_df.set_index("models", inplace=True)
tox21_df.set_index("models", inplace=True)
esol_df.set_index("models", inplace=True)
lipo_df.set_index("models", inplace=True)
freesolv_df.set_index("models", inplace=True)

#Create function to calculate p-value from rank-sum tests and save data in the format that can be used to create heatmap
def create_p_values_dict(data_df, metrics):
    baseline_models = data_df.index

    p_values_dict = {metric: pd.DataFrame(index=baseline_models, columns=baseline_models) for metric in metrics}

    for metric in metrics:
        for model1 in baseline_models:
            for model2 in baseline_models:
                #statistic, p_value = stats.wilcoxon(model1_scores, model2_scores, alternative='two-sided')
                stat, p_val = ranksums(
                    data_df.loc[model1, [f"{metric}1", f"{metric}2", f"{metric}3"]],
                    data_df.loc[model2, [f"{metric}1", f"{metric}2", f"{metric}3"]])
                p_values_dict[metric].at[model1, model2] = p_val

    for metric in metrics:
        p_values_dict[metric] = p_values_dict[metric].astype(float)

    p_values_binary_dict = {metric: p_values_dict[metric].applymap(lambda x: 1 if x < 0.05 else 0) for metric in metrics}
    
    return p_values_binary_dict

#create p-values binary dictionaries for each dataset
bbbp_p_values_binary_dict = create_p_values_dict(bbbp_df, classification_metrics)
bace_p_values_binary_dict = create_p_values_dict(bace_df, classification_metrics)
clintox_p_values_binary_dict = create_p_values_dict(clintox_df, classification_metrics)
sider_p_values_binary_dict = create_p_values_dict(sider_df, classification_metrics)
tox21_p_values_binary_dict = create_p_values_dict(tox21_df, classification_metrics)
esol_p_values_binary_dict = create_p_values_dict(esol_df, regression_metrics)
lipo_p_values_binary_dict = create_p_values_dict(lipo_df, regression_metrics)
freesolv_p_values_binary_dict = create_p_values_dict(freesolv_df, regression_metrics)


#define metrics for classification and regression
classification_metrics = ["roc", "pr"]
regression_metrics = ["rmse", "correlation"]

#set the font to Time for the heatmap plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18})

#adjust the figure layout to improve the spacing between plots
fig, axes = plt.subplots(2, 8, figsize=(22, 10))

#define the colormap and normalization
cmap = ListedColormap(['#8a8686', '#a84242'])
norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1], ncolors=2)

#generate heatmap plots for significance test
datasets = ["BBBP", "BACE", "ClinTox", "SIDER", "Tox21", "ESOL", "Lipo", "FreeSolv"]
p_values_binary_dicts = [bbbp_p_values_binary_dict, bace_p_values_binary_dict, clintox_p_values_binary_dict, sider_p_values_binary_dict, tox21_p_values_binary_dict, esol_p_values_binary_dict, lipo_p_values_binary_dict, freesolv_p_values_binary_dict]

for dataset_index, (dataset, p_values_binary_dict) in enumerate(zip(datasets, p_values_binary_dicts)):
    if dataset in ["BBBP", "BACE", "ClinTox", "SIDER", "Tox21"]:
        metrics = classification_metrics
        metric_titles = ["ROC-AUC", "PR-AUC"]
    else:
        metrics = regression_metrics
        metric_titles = ["RMSE", "Correlation"]
    
    for metric_index, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[metric_index, dataset_index]
        sns.heatmap(p_values_binary_dict[metric], annot=False, cmap=cmap, norm=norm, cbar=False, linewidths=1, linecolor='white', ax=ax)
        ax.set_title(f"{dataset}: {title}", fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_aspect('equal', adjustable='box')

#custom legend
legend_handles = [
    plt.Rectangle((0,0),1,1, color='#a84242', label='p-value < 0.05'),
    plt.Rectangle((0,0),1,1, color='#8a8686', label='p-value >= 0.05'),
]

fig.legend(handles=legend_handles, loc='lower center', ncol=2)

plt.tight_layout(rect=[0, 0.03, 1, 1])

plt.savefig('file_path', dpi=300, bbox_inches='tight')