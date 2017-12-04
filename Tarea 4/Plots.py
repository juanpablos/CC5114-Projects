import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.style.use('ggplot')


def make_plot(file_name):
    fn = file_name[:-4]

    with open(fn + "_info.txt") as f:
        target_number = next(f).split(':')[1].strip()

    data = pd.read_csv(file_name)
    ax = data.plot(kind='line', y='best_fitness')
    data.plot(kind='line', y='avg_value', ax=ax, style='--', secondary_y=True)
    plt.title("Fitness curve for {}".format(target_number))
    ax.set_ylabel("Fitness")
    plt.ylabel("Average fitness")
    ax.set_xlabel("Generations")
    figure = plt.gcf()
    figure.set_size_inches(19, 10)
    plt.savefig('Plots/{}'.format(fn.split("/")[1]), bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == "__main__":
    _file = "Results/gp_out_1.csv"
    make_plot(_file)
