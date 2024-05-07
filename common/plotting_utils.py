from pathlib import Path
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_scores(log_dir,
               subdir="scores",
               only_longest=False,
               min_length=-1,
               cumulative=False):
    longest = 0
    print(log_dir)
    overall_scores = []
    glob_pattern = log_dir + "/" + subdir + "/" + "*.pkl"
    for score_file in glob.glob(glob_pattern):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)
        if only_longest:
            if len(scores) > longest:
                overall_scores = [scores]
                longest = len(scores)
        else:
            if len(scores) >= min_length:  # -1 always passes!
                overall_scores.append(scores)

    if len(overall_scores) == 0:
        raise Exception("No scores in " + log_dir)

    min_length = min(len(s) for s in overall_scores)

    overall_scores = [s[:min_length] for s in overall_scores]

    score_array = np.array(overall_scores)
    # print(score_array.shape)
    if cumulative:
        score_array = np.cumsum(score_array, axis=1)
    return score_array


def get_plot_params(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    N = array.shape[0]
    top = means + (std / np.sqrt(N))
    bot = means - (std / np.sqrt(N))
    return median, means, top, bot


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def smoothen_data(scores, n=10):
    print(scores.shape)
    smoothened_cols = scores.shape[1] - n + 1
    
    smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
    for i in range(scores.shape[0]):
        smoothened_data[i, :] = moving_average(scores[i, :], n=n)
    return smoothened_data


def generate_plot(score_array, label, smoothen=False, experiment_name=None):
    if smoothen:
        score_array = smoothen_data(score_array, n=10)
    median, mean, top, bottom = get_plot_params(score_array)

    color_legend_map = {
        "vanilla": "dimgrey",
        "noisy": "firebrick",
        "per": "dodgerblue",
        "double": "blueviolet",
        "distributional": "darkorange",
        "dueling": "limegreen",
        "multi-step": "khaki"
    }

    x1 = list(range(len(mean)))

    if experiment_name in (["ant", "cheetah", "humanoid", "pendulum"]):
        x1 = [((i)*(10)*(200/193)) for i in x1]
    else:
        x1 = [((i)*10*(100/91)) for i in x1]


    #x1 = [((i)*10) for i in x1]

    if label in color_legend_map: 
        plt.plot(x1, mean, linewidth=2, label=label, alpha=0.9, color=color_legend_map[label])
    else:
        plt.plot(x1, mean, linewidth=2, label=label, alpha=0.9)       

    x2 = list(range(len(top)))
    
    #x2 = [((i)*10*(100/91)) for i in x2]
    

    if experiment_name in (["ant", "cheetah", "humanoid", "pendulum"]):
        x2 = [((i)*(10)*(200/193)) for i in x2]
    else:
        x2 = [((i)*10) for i in x2]

     
    if label in color_legend_map:
        plt.fill_between(x2, top, bottom, alpha=0.2, color=color_legend_map[label])
    else:
        plt.fill_between(x2, top, bottom, alpha=0.2)

    if experiment_name in (["ant", "cheetah", "humanoid"]):
        plt.xlim(0, 2000)
    elif experiment_name in (["lunarlander"]):
        plt.xlim(0, 200)
    elif experiment_name in (["pendulum"]):
        plt.xlim(0, 100)
    elif experiment_name in (["bipedalwalker", "hopper"]):
        plt.xlim(0, 1000)
    

def get_all_run_titles(experiment_name):
    parent = Path(experiment_name)
    run_titles = [d.name for d in parent.iterdir()]
    print(run_titles)
    return run_titles


def get_all_runs_and_subruns(experiment_name):
    parent = Path(experiment_name)
    run_titles = []
    for d in parent.iterdir():
        if d.is_dir():
            run_titles.append(d.name)
            for sub_d in d.iterdir():
                if sub_d.is_dir():
                    run_titles.append(f"{d.name}/{sub_d.name}")
    return run_titles
