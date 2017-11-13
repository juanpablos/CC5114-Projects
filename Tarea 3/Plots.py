import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.style.use('ggplot')


def make_plot(file_name):
    data = pd.read_csv(file_name)
    ax = data.plot(kind='line', y='best_fitness', ylim=(0., 1.05))
    data.plot(kind='line', y='avg_value', ax=ax, style='--')
    plt.title("Neural network accuracy by number of generations")
    figure = plt.gcf()
    figure.set_size_inches(19, 10)
    plt.savefig('Images/{}'.format(file_name[:-3]), bbox_inches='tight', dpi=200)
    plt.close()


_file = "network_out_1000-03-50-highprob.csv"
make_plot(_file)
