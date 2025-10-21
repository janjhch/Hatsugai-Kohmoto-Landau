import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

textwidth = 6.377953


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
def many_plots(label_array: list, x_arrays: list, y_arrays: list, xlabel: str, ylabel: str, title='', ymax=None, ymin=None, points=False):
    plt.figure(dpi=150)
    # Plot erstellen
    if points == False:
        for i in range(len(x_arrays)):
            plt.plot(x_arrays[i], y_arrays[i], linestyle='-', label=label_array[i])
    else:
        for i in range(len(x_arrays)):
            plt.plot(x_arrays[i], y_arrays[i], 'o', label=label_array[i], ms=0.5)
    
    # Achsenbeschriftungen und Titel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ymax != None:
        plt.ylim(top=ymax)
    if ymin != None:
        plt.ylim(bottom=ymin)
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




def thesis_plot_one_line(ax, xarray: np.ndarray, yarray: np.ndarray, x_label: str, y_label: str, title='', yticks=2, ylim_diff=0, ymax=None):
    # Main curve
    #ax.plot(xarray, yarray, color="black", linewidth=1)
    ax.plot(xarray, yarray, linewidth=1)

    # Achsenbeschriftungen und Titel
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    ax.set_xlim(np.min(xarray), np.max(xarray))
    if ymax == None:
        ax.set_ylim(np.min(yarray)- ylim_diff, np.max(yarray))
    else:
        ax.set_ylim(np.min(yarray)- ylim_diff, ymax)

    #ax.tick_params(top=True, right=True, direction='in', pad=7)
    ax.tick_params(top=False, right=False, bottom=False, left=False)
    # Set major tick intervals
    #ax.xaxis.set_major_locator(MultipleLocator(0.25))  # x-axis ticks every 0.2 units
    ax.yaxis.set_major_locator(MultipleLocator(yticks))  # y-axis ticks every 0.5 units

    if title != '':
        ax.set_title(title, fontsize=12, weight='bold')

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

def thesis_doubleplot_one_line(xarrays: list, yarrays: list, xlabel:str, ylabel:str, titles=['',''], yticks=[2,2], ylimits=[0,0], save_title='', ymax=None):

    # === Combined Figure ===
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(textwidth, 4.5 * textwidth / 10)

    thesis_plot_one_line(axes[0], xarrays[0], yarrays[0], xlabel, ylabel, titles[0], yticks[0], ylimits[0], ymax)
    thesis_plot_one_line(axes[1], xarrays[1], yarrays[1], xlabel, ylabel, titles[1], yticks[1], ylimits[1], ymax)
    """
        # --- Add subplot labels (a) and (b) ---
        labels = ['(a)', '(b)']
        for ax, label in zip(axes, labels):
            ax.text(
                0.02, 0.98, label, transform=ax.transAxes,
                fontweight='bold', va='top', ha='left'
            )
    """
    plt.tight_layout(rect=[0,0,1,1])

    if save_title != '':
        plt.savefig(save_title, dpi=1000, bbox_inches="tight")
        
    plt.show()


def thesis_plot_multiple_lines(ax, label_array: list, x_arrays: list, y_arrays: list, xlabel: str, ylabel: str, title='', yticks=2, ylim_diff=0, ymax=None,
                               reverse=True):

    # Plot erstellen
    if reverse==True:
        for i in reversed(range(len(x_arrays))):
            ax.plot(x_arrays[i], y_arrays[i], linestyle='-', label=label_array[i], linewidth=1)
    else:
        for i in range(len(x_arrays)):
            ax.plot(x_arrays[i], y_arrays[i], linestyle='-', label=label_array[i], linewidth=1)
    
    # Achsenbeschriftungen und Titel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    ax.set_xlim(np.min(x_arrays), np.max(x_arrays))
    if ymax == None:
        ax.set_ylim(np.min(y_arrays)- ylim_diff, np.max(y_arrays))
    else:
        ax.set_ylim(np.min(y_arrays)- ylim_diff, ymax)

    ax.tick_params(top=False, right=False, bottom=False, left=False)
    ax.set_xlim(0, 2)

    #ax.legend(loc='best')
    if title != '':
        ax.set_title(title, weight='bold')

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

    ax.yaxis.set_major_locator(MultipleLocator(yticks))  # y-axis ticks every 0.5 units
    

def thesis_doubleplot_multiple_lines(label_arrays:list, xarrays: list, yarrays: list, xlabel:str, ylabels:list,
                                      titles=['',''], yticks=[2,2], ylimits=[0, 0], ymax=[None, None], legend=True,
                                       reverse=True, save_title=''):

    # === Combined Figure ===
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(textwidth, 4.5 * textwidth / 10)

    thesis_plot_multiple_lines(axes[0], label_arrays[0], xarrays[0], yarrays[0], xlabel, ylabels[0], titles[0], yticks[0], ylimits[0], ymax[0], reverse)
    thesis_plot_multiple_lines(axes[1], label_arrays[1], xarrays[1], yarrays[1], xlabel, ylabels[1], titles[1], yticks[1], ylimits[1], ymax[1], reverse)

    # Collect from both axes
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    """
        # --- Add subplot labels (a) and (b) ---
        plot_labels = ['(a)', '(b)']
        for ax, plot_label in zip(axes, plot_labels):
            ax.text(
                0.02, 0.98, plot_label, transform=ax.transAxes,
                fontweight='bold', va='top', ha='left'
            )
    """
    # Keep only occupation categories, and remove duplicates
    keep = label_arrays[0]
    handle_label_dict = {l: h for h, l in zip(handles, labels) if l in keep}

    if legend == True:
        # Global legend
        fig.legend(
            handle_label_dict.values(),
            handle_label_dict.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),  # move legend below plots
            ncol=4,
            frameon=False
        )


    plt.tight_layout(rect=[0,0,1,1])

    if save_title != '':
        plt.savefig(save_title, dpi=1000, bbox_inches="tight")
        
    plt.show()




def thesis_plot_pd(rho: np.array, U_c_norm: np.array, save_title=''):
    # Create the plot
    fig, ax = plt.subplots()

    fig.set_size_inches(6.377953 / 1.5, 2.73341 * 1.2)

    ax.plot(rho, U_c_norm, label=r'$U_c (\rho)$', color='black', linewidth=1)

    ax.vlines(x=1, ymin=1, ymax=1.6, color='black', linestyle='-')

    ax.text(1, 0.4, 'I', ha='center')
    ax.text(0.4, 1.0, 'II', ha='center')
    ax.text(1.6, 1.0, 'III', ha='center')
    ax.text(1.2, 1.25, 'IV', ha='left', va='center')

    ax.annotate(
    '', 
    xy=(1.0, 1.25),    # arrowhead (end point)
    xytext=(1.2, 1.25),# tail (start point)
    arrowprops=dict(
        arrowstyle='->',   # simple one-sided arrow
        color='black',
        linewidth=1
    )
)

    # Labels and legend
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$U/W$', rotation=0, va='top')

    ax.legend(frameon=False, loc='upper left')

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.6)
    ax.tick_params(top=True, right=True, direction='in', pad=7)
    # Set major tick intervals
    ax.xaxis.set_major_locator(MultipleLocator(0.5))  # x-axis ticks every 0.2 units
    ax.yaxis.set_major_locator(MultipleLocator(0.5))  # y-axis ticks every 0.5 units

    if save_title != '':
        plt.savefig(save_title, dpi=1000, bbox_inches="tight")

    plt.show()

def pdplot(ax, rho: np.array, U_c_norm: np.array, title:str):
    ax.plot(rho, U_c_norm, label=r'$U_c (\rho)$', color='black', linewidth=1)

    ax.vlines(x=1, ymin=1, ymax=1.6, color='black', linestyle='-')

    ax.text(1, 0.4, 'I', ha='center')
    ax.text(0.4, 1.0, 'II', ha='center')
    ax.text(1.6, 1.0, 'III', ha='center')
    ax.text(1.2, 1.25, 'IV', ha='left', va='center')

    ax.annotate(
    '', 
    xy=(1.0, 1.25),    # arrowhead (end point)
    xytext=(1.2, 1.25),# tail (start point)
    arrowprops=dict(
        arrowstyle='->',   # simple one-sided arrow
        color='black',
        linewidth=1
    )
)

    # Labels and legend
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$U/W$', rotation=0, va='top')

    ax.legend(frameon=False, loc='upper left')

    ax.set_title(title)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.6)
    ax.tick_params(top=True, right=True, direction='in', pad=7)
    # Set major tick intervals
    ax.xaxis.set_major_locator(MultipleLocator(0.5))  # x-axis ticks every 0.2 units
    ax.yaxis.set_major_locator(MultipleLocator(0.5))  # y-axis ticks every 0.5 units



def thesis_doubleplot_pd(rho_arrays: list, U_c_arrays: list, titles: list, save_title=''):

    # === Combined Figure ===
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(textwidth, 4.5 * textwidth / 10)

    pdplot(axes[0], rho_arrays[0], U_c_arrays[0], titles[0])
    pdplot(axes[1], rho_arrays[1], U_c_arrays[1], titles[1])

    plt.tight_layout(rect=[0,0,1,1])

    if save_title != '':
        plt.savefig(save_title, dpi=1000, bbox_inches="tight")
        
    plt.show()


def thesis_singleplot_multiple_lines(label_array: list, x_arrays: list, y_arrays: list, xlabel: str, ylabel: str, title='', yticks=2, save_title=''):

    # Create the plot
    fig, ax = plt.subplots()

    fig.set_size_inches(6.377953 / 1.7, 2.73341 * 1.0)

    # Plot erstellen
    for i in reversed(range(len(x_arrays))):
        ax.plot(x_arrays[i], y_arrays[i], linestyle='-', label=label_array[i], linewidth=1)
    
    # Achsenbeschriftungen und Titel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)

    ax.tick_params(top=False, right=False, bottom=False, left=False)
    ax.set_xlim(0, 2)

    #ax.legend(loc='best')
    if title != '':
        ax.set_title(title, weight='bold',
                     loc="center",
                     )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

    ax.yaxis.set_major_locator(MultipleLocator(yticks))  # y-axis ticks every 0.5 units
    
    ax.legend(loc="center left",
            bbox_to_anchor=(1, 0.5),  # move legend below plots
            ncol=1,
            frameon=False)

    if save_title != '':
        plt.savefig(save_title, dpi=1000, bbox_inches="tight")

    # --- after you finish building the figure (plot, legend, etc.) ---
    fig.canvas.draw()          # ensure renderer exists and artists have positions
    renderer = fig.canvas.get_renderer()

    # 1) Canvas nominal size (the size you set via set_size_inches)
    canvas_w_in, canvas_h_in = fig.get_size_inches()
    print(f"Canvas size (set)     : {canvas_w_in:.3f} in x {canvas_h_in:.3f} in "
        f"= {canvas_w_in*2.54:.2f} cm x {canvas_h_in*2.54:.2f} cm")

    # 2) Tight bounding box: the actual drawn extents that 'bbox_inches=tight' would use
    tight_bbox = fig.get_tightbbox(renderer)   # Bbox in display units (pixels or points)
    # convert display units -> inches by dividing by figure dpi
    tight_w_in = tight_bbox.width 
    tight_h_in = tight_bbox.height
    print(f"Tight bbox (rendered)  : {tight_w_in:.5f} in x {tight_h_in:.5f} in "
        f"= {tight_w_in*2.54:.2f} cm x {tight_h_in*2.54:.2f} cm")
            
    plt.show()
