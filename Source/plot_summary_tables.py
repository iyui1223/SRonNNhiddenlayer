import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def plot_table(df, title, filename):
    pivot_df = df.pivot(index='ForceType', columns='Model', values=title)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": title})
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', required=True, help='Path to sparsity table TSV')
    parser.add_argument('--r2', required=True, help='Path to R2 score table TSV')
    parser.add_argument('--outdir', default="Work", help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sparsity_df = pd.read_csv(args.sparsity, sep="\t")
    r2_df = pd.read_csv(args.r2, sep="\t")

    plot_table(sparsity_df, "Sparsity", os.path.join(args.outdir, "sparsity_table.png"))
    plot_table(r2_df, "R2", os.path.join(args.outdir, "r2_table.png"))
