from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def show_txs(dataset: Dataset, indexes, txs, figsize=(3, 10)):
    nrows = len(indexes)
    ncols = 1 + len(txs)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if nrows == 1:
        axs = [axs]
    if ncols == 1:
        axs = [[ax] for ax in axs]

    for row, index in enumerate(indexes):
        # manually plot original
        X, y = dataset[index]
        axs[row][0].imshow(X, cmap="Greys_r")
        axs[row][0].set_xticks([])
        axs[row][0].set_yticks([])
        if row == 0:
            axs[row][0].set_title("Original", fontsize=10)

        for n_tx, col in enumerate(range(1, ncols)):
            if row == 0:
                label = getattr(txs[n_tx], "__name__", f"tx_n_{n_tx}")
                axs[row][col].set_title(label, fontsize=10)
            axs[row][col].imshow(txs[n_tx](X), cmap="Greys_r")
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])
    fig.tight_layout()
    return fig
