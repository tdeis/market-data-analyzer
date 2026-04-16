import matplotlib.pyplot as plt


def plot_data(data, title, ylabel="Value"):
    ax = data.plot(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.show()
