import matplotlib.pyplot as plt

def save_to_file(filename):
    plt.savefig('./figures/' + filename + '.svg')

# Useage:
# import save_fig as sf
# sf.save_to_file('stacked-bar')