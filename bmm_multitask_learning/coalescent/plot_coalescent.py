import matplotlib.pyplot as plt


def plot_coalesce(y_history, pairs):
    fig, ax = plt.subplots(1, 1)
    old_t = 0
    for i, layer in enumerate(y_history):
        new_item = min(layer, key=lambda x: x.t)
        max_t = new_item.t
        for item in layer:
            ax.plot([max_t], item.mean[0], ".")
        if i > 0:

            # рисуем предков
            predecessors = pairs[i-1]
            for predecessor in predecessors:
                src = y_history[i-1][predecessor]
                x, y, dx, dy = old_t, src.mean[0], \
                    max_t - old_t, \
                    -(src.mean - new_item.mean)[0]
                
                ax.arrow(x, y, dx, dy, width=-dx/1000)
            for j in range(len(layer) + 1):
                if j in predecessors:
                    continue
                src = y_history[i-1][j]
                x, y, dx, dy = max_t, src.mean[0], -(max_t - old_t), 0
                ax.arrow(x, y, dx, dy, width=-dx/1000)

        old_t = max_t
    return fig

