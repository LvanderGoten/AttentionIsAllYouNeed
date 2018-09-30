import os
import numpy as np
from argparse import ArgumentParser
from tqdm import trange


def main():
    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--embedding", required=True)
    parser.add_argument("--limit", type=int)

    # Parse
    args = parser.parse_args()

    with open(args.embedding, "r") as f:
        num_embeddings, num_dimensions = next(f).strip().split(" ")
        num_embeddings = int(num_embeddings) if not args.limit else args.limit
        num_dimensions = int(num_dimensions)
        arr = np.zeros(shape=[num_embeddings + 2, num_dimensions], dtype=np.float32)
        tokens = np.full(shape=[num_embeddings + 2], fill_value="", dtype=np.object)

        # Special symbols (have some random embedding)
        tokens[0] = "<<START>>"
        tokens[1] = "<<END>>"
        arr[0, :] = np.random.randn(num_dimensions)
        arr[1, :] = np.random.randn(num_dimensions)

        for i in trange(2, num_embeddings + 2):
            line = next(f)
            items = line.rstrip().split(" ")
            token = items[0]
            embedding = np.array(items[1:], dtype=np.float32)
            arr[i, :] = embedding
            tokens[i] = token

    # Save
    fid = os.path.join(os.path.dirname(args.embedding), os.path.splitext(os.path.basename(args.embedding))[0])
    np.save(file="{}_embeddings.npy".format(fid), arr=arr)
    with open("{}_tokens.txt".format(fid), encoding="utf-8", mode="w") as f:
        f.write("\n".join(tokens.tolist()))


if __name__ == "__main__":
    main()