import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def plot_table(df, title, filename):
    pivot_df = df.pivot(index='ForceType', columns='Model', values=title)
    plt.figure(figsize=(6, 4))

    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",            # Pale colormap
        vmin=0, vmax=1,
        cbar=False,
        annot_kws={"fontsize": 12, "color": "black"}
    )

    # Apply transparency to the heatmap (alpha)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    for t in ax.collections:
        t.set_alpha(0.3)  # Reduce heatmap color intensity

    plt.title(title, fontsize=14)
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', required=True, help='Path to sparsity table TSV')
    parser.add_argument('--r2', required=True, help='Path to R2 score table TSV')
    parser.add_argument('--acc', required=True, help='Path to ACC table TSV')
    parser.add_argument('--outdir', default="Work", help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sparsity_df = pd.read_csv(args.sparsity, sep="\t")
    r2_df = pd.read_csv(args.r2, sep="\t")
    acc_df = pd.read_csv(args.acc, sep="\t")

    plot_table(sparsity_df, "Sparsity", os.path.join(args.outdir, "sparsity_table.png"))
    plot_table(r2_df, "R2", os.path.join(args.outdir, "r2_table.png"))
    plot_table(acc_df, "ACC", os.path.join(args.outdir, "acc_table.png"))
