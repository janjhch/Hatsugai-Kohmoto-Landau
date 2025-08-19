import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Simple plot function for quickly checking results
def one_plot(x_array: np.ndarray, y_array: np.ndarray, x_label: str, y_label: str, title=''):
    plt.figure(dpi=150)
    # Plot erstellen
    plt.plot(x_array, y_array, linestyle='-')
    
    # Achsenbeschriftungen und Titel
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.title(title)
    plt.show()


# Create two plots next to each other in one figure
def two_plots_one_figure(x_array_1: np.ndarray, x_array_2: np.ndarray, y_array_1: np.ndarray, y_array_2: np.ndarray, 
                         title: str, y_label: str, title_1='', title_2=''):
    # Create side-by-side subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=150)
    plt.style.use('seaborn-v0_8-pastel')
    
    # Plot 1
    axes[0].plot(x_array_1, y_array_1)
    axes[0].set_title(title_1)
    axes[0].set_xlabel(r'$\rho$')
    axes[0].set_ylabel(y_label)
    axes[0].grid(True)
    
    # Plot 2
    axes[1].plot(x_array_2, y_array_2)
    axes[1].set_title(title_2)
    axes[1].set_xlabel(r'$\rho$')
    axes[1].set_ylabel(y_label)
    axes[1].grid(True)
    
    fig.suptitle(title, weight='bold')
    
    # Layout adjustment
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)  # increase this value for more space
    plt.show()


# Two graphs in one plot
def two_graphs_in_one_plot(x_array_1: np.ndarray, x_array_2: np.ndarray, y_array_1: np.ndarray, y_array_2: np.ndarray, 
                           label_1: str, label_2: str, x_label: str, y_label: str, title=''):
    plt.figure(dpi=150)
    #plt.style.use('seaborn-v0_8-muted')
    
    # Plot erstellen
    plt.plot(x_array_1, y_array_1, linestyle='-', label=label_1)
    plt.plot(x_array_2, y_array_2, linestyle='-', label=label_2)
    
    # Achsenbeschriftungen und Titel
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, weight='bold')
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.grid(True)
    plt.show()

# Plots any number of graphs in one plot, arrays must be given inside one list for x and y respectively
def many_plots(label_array: list, x_arrays: list, y_arrays: list, xlabel: str, ylabel: str, title=''):
    plt.figure(dpi=150)
    # Plot erstellen
    for i in range(len(x_arrays)):
        plt.plot(x_arrays[i], y_arrays[i], linestyle='-', label=label_array[i])
    
    # Achsenbeschriftungen und Titel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


# Plot the Phase Diagram, x_values should be an rho array, y_values the normalized critical interaction
def plot_phase_diagram(x_values: np.ndarray, y_values: np.ndarray, x_label: str, y_label: str, y_max: float, 
                       title='none', legend=False, fill=False, alph=1):    
    # Create the plot
    fig, ax = plt.subplots()
    
    # Plot the dividing curve
    ax.plot(x_values, y_values, label=r'$U_c$', color='black')
    
    # Plot the vertical line at x = 1, starting from y=1 upwards
    ax.vlines(x=1, ymin=1, ymax=y_max, color='black', linestyle='-', label='MIT')
    
    # Fill areas for visualization (optional)
    if fill:
        ax.fill_between(x_values, y_values, y_max, where=(x_values < 1), 
                        interpolate=True, color='#DF1728', alpha=alph, label='Area 1')
        
        ax.fill_between(x_values, y_values, y_max, where=(x_values > 1), 
                        interpolate=True, color='#4E3DE1', alpha=alph, label='Area 2')
        
        ax.fill_between(x_values, 0, y_values, color='#B11CC2',
                         alpha=alph, label='Area 3 (Above Curve)')
    
    ax.plot(1, 1, marker='o', color='black', markersize=6, label='QCP')
    
    ax.text(1, 0.9, 'QCP', fontsize=12, color='black', ha='center')
    ax.text(0.25, 1.3, 'Only singly \noccupied states', fontsize=12, color='black', ha='left')
    ax.text(1.25, 1.3, 'All states occupied\nNo holes', fontsize=12, color='black', ha='left')
    ax.text(1, 0.4, 'Holes, singly and\n doubly occupied states', fontsize=12, color='black', ha='center')
    ax.text(0.48, 0.52, r'$U_c$', fontsize=12, color='black', ha='right')
    ax.text(1.52, 0.52, r'$\overline{U}_c$', fontsize=12, color='black', ha='left')
    
    # Labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title != 'none':
        ax.set_title(title)
    if legend == True:
        ax.legend()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, y_max)
    ax.tick_params(top=True, right=True, direction='in', pad=7)
    # Set major tick intervals
    ax.xaxis.set_major_locator(MultipleLocator(0.25))  # x-axis ticks every 0.2 units
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # y-axis ticks every 0.5 units

    plt.show()
   