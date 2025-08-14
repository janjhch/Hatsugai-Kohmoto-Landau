import matplotlib.pyplot as plt

# Simple plot function for quickly checking results
def one_plot(x_array, y_array, x_label, y_label, title):
    plt.figure(dpi=200)
    # Plot erstellen
    plt.plot(x_array, y_array, linestyle='-')
    
    # Achsenbeschriftungen und Titel
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.title(title)
    plt.show()


# Create two plots next to each other in one figure
def two_plots_one_figure(x_array_1, x_array_2, y_array_1, y_array_2, title, y_label, title_1='', title_2=''):
    # Create side-by-side subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=200)
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
def two_graphs_in_one_plot(x_array_1, x_array_2, y_array_1, y_array_2, label_1, label_2, y_label, title=''):
    plt.figure(dpi=200)
    plt.style.use('seaborn-v0_8-muted')
    
    # Plot erstellen
    plt.plot(x_array_1, y_array_1, linestyle='-', label=label_1)
    plt.plot(x_array_2, y_array_2, linestyle='-', label=label_2)
    
    # Achsenbeschriftungen und Titel
    plt.xlabel(r'$\rho$')
    plt.ylabel(y_label)
    plt.title(title, weight='bold')
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.show()