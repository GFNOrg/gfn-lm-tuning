import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display, Markdown
import re
import unidecode
from rng.rng_utils import format_dict


def slugify(text):
    text = unidecode.unidecode(text).lower()
    return re.sub(r"[\W_]+", "-", text)


def plot_distribution(
    dist,
    n_max=512 * 8,
    model_name="Model",
    color="lightseagreen",
    xlims=None,
    number_of_NaNs=0,
    font="Times New Roman",
    filename="model"
):
    plt.rcParams["font.family"] = font

    display(Markdown(f"## {model_name}: Distribution of generated numbers"))

    sns.set_theme(style="whitegrid", font=font)

    bins = range(min(dist), max(dist) + 2)

    plot = sns.histplot(
        dist,
        color=color,
        linewidth=0,
        bins=bins,
        stat="probability"
    )

    # plot.set_title(
    #     f"Distribution of generated outcomes ({len(dist)} numbers, {number_of_NaNs} NaNs)"
    # )
    plot.set_xlabel("Number")
    # plot.set_ylabel("Empirical probability")
    plot.set_ylabel("")
    
    if xlims is not None:
        plot.set_xlim(xlims)

    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot.figure.savefig(f"plots/{slugify(model_name)}_{n_max}.pdf", bbox_inches="tight")
    plt.show()


def visualize(df):
    sns.set_style("whitegrid")
    for dist in df["Distribution"].unique():
        for params in df[df["Distribution"] == dist]["Parameters"].unique():
            subset = df[(df["Distribution"] == dist) & (df["Parameters"] == params)]
            plt.figure(figsize=(10, 6))

            if (subset["Data Type"] == "int").all():
                sns.histplot(
                    subset["Value"], bins=len(subset["Value"].unique()), kde=False
                )
            else:
                sns.histplot(subset["Value"], kde=True)

            plt.title(f"Histogram of {dist} with parameters {format_dict(params)}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()


def plot_kl_divergence(results):
    for model, datasets in results.items():
        plt.figure(figsize=(10, 6))
        names = list(datasets.keys())
        values = list(datasets.values())

        sns.barplot(x=names, y=values)
        plt.title(f'KL Divergence for {model}')
        plt.ylabel('KL Divergence')
        plt.show()