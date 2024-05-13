import numpy as np
import torch.nn.functional as F
from scipy.signal import normalize
import argparse
import os


def neg_recall(mat, k_value):
    """
    This function prints a table with the given title and metrics.

    Parameters:
    title (str): The title of the table.
    metrics (dict): A dictionary where keys are metric names and values are metric values.

    Returns:
    None
    """
    neg_lists = []
    N = len(mat)

    # For each row in the matrix...
    for i in range(N):
        array = np.arange(N)
        np.random.shuffle(array)
        neg_list = list(array[:32])
        # If the current row index is in the negative list, remove it.
        if i in neg_list:
            neg_list.remove(i)
        else:
            neg_list.pop()
        # Append the negative list to the list of negative lists.
        neg_lists.append(neg_list)

    # Initialize a counter for the number of hits.
    hits = 0

    # For each row in the matrix...
    for rowid in range(len(mat)):
        row = mat[rowid]
        negsocres = list(row[neg_lists[rowid]])
        count_large = 0

        # For each score in the negative scores...
        for one_score in negsocres:
            # If the score at the current row index is less than this score, increment the counter.
            if row[rowid] < one_score:
                count_large += 1

        # If the number of scores that are larger is less than or equal to k_value - 1, increment the hits counter.
        if count_large <= k_value - 1:
            hits += 1

    # Return the number of hits.
    return hits


def main(args):
    """
    This function is the main entry point of the script. It loads embeddings, calculates similarities,
    and prints recall metrics.

    Parameters:
    args (argparse.Namespace): The command-line arguments.

    Returns:
    None
    """

    # Retrieve the list of experiment directories, retrieval type, and protocol from the command-line arguments.
    expdirs = args.expdirs
    retrieval_type = args.retrieval_type
    protocal = args.protocal

    # Define a list of values for K (the number of top elements to consider in the recall calculation).
    K_list = [1, 2, 3, 5, 10]

    # Initialize a list of lists to store the recall values for each experiment directory.
    RecK_list = [[] for i in expdirs]

    # For each experiment directory...
    for index in range(len(expdirs)):
        # Retrieve the current experiment directory.
        exp_dir = expdirs[index]

        # Set the directory containing the embeddings to the experiment directory.
        emb_dir = exp_dir

        # Define the paths to the motion, text, and SBERT embeddings.
        motion_emb_dir = os.path.join(emb_dir, "motion_embedding.npy")
        text_emb_dir = os.path.join(emb_dir, "text_embedding.npy")
        sbert_emb_dir = os.path.join(emb_dir, "sbert_embedding.npy")

        # Load the embeddings from the files.
        text_embedding = np.load(text_emb_dir)
        motion_embedding = np.load(motion_emb_dir)
        sbert_embedding = np.load(sbert_emb_dir)

        # Normalize the SBERT embeddings.
        sbert_embedding = sbert_embedding / np.linalg.norm(
            sbert_embedding, axis=1, keepdims=True
        )

        # Calculate the text-to-motion and motion-to-text similarity matrices.
        T2M_logits = text_embedding @ (motion_embedding.T)
        M2T_logits = motion_embedding @ (text_embedding.T)

        # Depending on the retrieval type, select the appropriate similarity matrix.
        if retrieval_type == "T2M":
            logits_matrix = T2M_logits
        elif retrieval_type == "M2T":
            logits_matrix = M2T_logits

        # Calculate the SBERT similarity matrix.
        sbert_sim = sbert_embedding @ (sbert_embedding.T)
        N = sbert_embedding.shape[0]

        # Initialize a list to store the target lists.
        target_list = []

        # If the protocol is A or B...
        if protocal == "A" or protocal == "B":
            for i in range(N):
                target_list_i = []
                for j in range(N):
                    # If the protocol is A and the other embedding is the same as the current one, add it to the target list.
                    if protocal == "A":
                        if j == i:
                            target_list_i.append(j)
                    # If the protocol is B and the SBERT similarity between the other embedding and the current one is at least 0.9, add it to the target list.
                    elif protocal == "B":
                        if sbert_sim[i][j] >= 0.9:
                            target_list_i.append(j)

                # Add the target list for this embedding to the list of target lists.
                target_list.append(target_list_i)

            # Sort the indices of the embeddings in the similarity matrix in descending order of similarity.
            sorted_embedding_idx = np.argsort(-logits_matrix, axis=1)
            i = 0
            for k in K_list:
                hits = 0
                for i in range(N):
                    # Get the top K embeddings in the sorted list.
                    pred = list(sorted_embedding_idx[i][:k])
                    # If any of the top K embeddings are in the target list for this embedding, increment the hits counter.
                    for item in pred:
                        if item in target_list[i]:
                            hits += 1
                            break
                # Calculate the recall for this value of K and add it to the list of recall values for this experiment directory.
                RecK_list[index].append("%.3f" % (100.0 * (hits / N)))
                i += 1

        # If the protocol is D...
        elif protocal == "D":
            for k in K_list:
                # Calculate the negative recall for this value of K and add it to the list of recall values for this experiment directory.
                hits = neg_recall(logits_matrix, k)
                RecK_list[index].append("%.3f" % (100.0 * (hits / N)))

    # To markdown table format
    print("|   Metrics   |", end="  ")
    for k in K_list:
        print(f"Recall @{k} |", end="  ")
    print()
    print("|-------------|", end="  ")
    for k in K_list:
        print("--------- |", end="  ")
    print()
    for l in range(len(RecK_list)):
        exp_name = expdirs[l].split("/")[-2]
        print(f"|{exp_name} |", end="  ")
        for item in RecK_list[l]:
            print(item, end="   |")
        print("")
    print()


if __name__ == "__main__":
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser()

    # Add arguments for retrieval type, protocol, and experiment directories
    parser.add_argument("--retrieval_type", default="T2M", type=str, help="T2M or M2T")
    parser.add_argument("--protocal", default="A", type=str, help="A, B, or D")
    parser.add_argument("--expdirs", nargs="+")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
