import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

font = {'family': 'serif',
        # 'serif': ['Computer Modern'],
        'serif': ['Helvetica'],
        # 'weight': 'bold',
        'size': 8}

savefig = {'bbox': 'tight',
           'pad_inches': 0.01,
           'dpi': 1200,
           'transparent': True}

matplotlib.rc('font', **font)
matplotlib.rc('savefig', **savefig)

OUTPUT_DIR = "_figures/model_evaluation/"
mm_to_inch = lambda val: np.array(val) * 0.0393701

randomization_lengths = {"T1": 5, "T2": 6, "T3": 7, "T4": 7}
data_cols = ["w/o", "Tc", "Neo", "both"]
bases = {"A": 0, "G": 1, "C": 2, "U": 3, "-": 4}

colors_ligands = {"w/o": "#A1CB79", "Tc": "#EBC84F", "Neo": "#5799D1", "both": "#99A3B5"}
color_score = "#E86420"
color_ucb = "#F5D67D"

reference_length = 7

color_orange = "#E86420"  # Orange shade
cmap_orange = LinearSegmentedColormap.from_list("OrangeMap", ["white", color_orange])
cmap_orange_black = LinearSegmentedColormap.from_list("OrangeBlackMap", ["white", color_orange, "#742E0C"])

color_blue = "#1067A9"  # Blue shade
cmap_blue = LinearSegmentedColormap.from_list("BlueMap", ["white", color_blue])

color_teal = "#44B4B4"  # Yellow shade
cmap_teal = LinearSegmentedColormap.from_list("TealMap", ["white", color_teal])
cmap_teal_black = LinearSegmentedColormap.from_list("TealBlackMap", ["white", color_teal, "#235D5D"])


color_red = "#BC4944"  # Orange shade
cmap_red = LinearSegmentedColormap.from_list("RedMap", ["white", color_red])
cmap_red_black = LinearSegmentedColormap.from_list("RedBlackMap", ["white", color_red, "#6C211B"])


cmap_blue_to_orange = LinearSegmentedColormap.from_list("BlueOrangeMap", [color_blue, "white", color_orange])
cmap_teal_to_orange = LinearSegmentedColormap.from_list("TealOrangeMap", [color_teal, "white", color_orange])
cmap_black_to_teal_to_orange = LinearSegmentedColormap.from_list("BlackTealOrangeMap", ["#235D5D", color_teal, "white", color_orange])

# Orange shade
cmap_score = LinearSegmentedColormap.from_list("ScoreMap", ["white", color_score])
cmap_ucb = LinearSegmentedColormap.from_list("UCBMap", ["white", color_ucb])