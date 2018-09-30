from argparse import ArgumentParser
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def main():
    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--de_tokens_fname", required=True)
    parser.add_argument("--en_tokens_fname", required=True)
    parser.add_argument("--attention_weights_fname", required=True)

    # Parse
    args = parser.parse_args()

    # Load
    de_tokens = [token.strip() for token in open(args.de_tokens_fname)]
    en_tokens = [token.strip() for token in open(args.en_tokens_fname)]
    attn_weights = np.load(args.attention_weights_fname)

    # Plot
    sns.set(font_scale=1.4)
    plt.gca().xaxis.tick_top()
    plt.gca().yaxis.tick_left()

    sns.heatmap(data=attn_weights, xticklabels=de_tokens, yticklabels=en_tokens, cbar=False, cmap="YlGnBu", vmin=0, vmax=1)
    plt.savefig("attention.png", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    main()