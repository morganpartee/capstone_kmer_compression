from Bio import SeqIO
from os import listdir
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix
from itertools import chain
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import ray
from datetime import datetime
from sys import argv
from random import sample
from functools import reduce


@ray.remote
def count_k_mers(seq: str, ksize: int, verbose=True, reverse_compliment_only=True):
    """
    This function counts each k-mer in a provided string.
    It returns a sorted dictionary of k-mer keys and count values.

    This function expects a tuple of index and the DNA sequence, so that it
    can be mapped with multiprocessing.

    Examples:
    dna = (1, 'acgtccagdn')
    In [14]: count_k_mers(dna, 3)
    Counted 1
    Out[14]: (1, {'acg': 2, 'agd': 1, 'cag': 1, 'cca': 1, 'gac': 1, 'gga': 1})

    In [15]: count_k_mers(dna, 5)
    Counted 1
    Out[15]: (1, {'acgtc': 1, 'ccagd': 1, 'ggacg': 1, 'gtcca': 1, 'tccag': 1})
    """

    def reverse_compliment(seq: str):
        """
        This function returns the reverse compliment of the dna string. n's are not changed.
        Strings are translated, then returned in reverse.
        Examples:
        >>> reverse_compliment('tacg')
        'cgta'
        >>> reverse_compliment('acgt')  # Yep, this is it's own reverse compliment
        'acgt'
        """
        return seq.translate({97: 116, 99: 103, 103: 99, 116: 97})[::-1]

    def lexical_score(seq: str):
        """
        This function calculates a 'lexical score' for the given string.
        This score is just the sum of the ord value for each character in the string.
        This is used to determine which version (forward or reverse compliment)
        of a k-mer to store.
        Examples:
        >>> lexical_score('aaaa')
        388
        >>> lexical_score('tttt')
        464
        """
        return sum(map(ord, seq))

    mers = Counter()
    for i in range(len(seq) - ksize):
        mer = seq[i : i + ksize]  # Each k-mer is a sliding window piece of the sequence
        if reverse_compliment_only:
            rc = reverse_compliment(mer)
            if lexical_score(mer) < lexical_score(rc):  # update the counter with:
                mers.update([mer])  # the  k-mer if it's lower lexically
            else:
                mers.update([rc])  # Else the reverse compliment
        else:
            mers.update([mer])

    if verbose:  # Let the nervous observer know it's doing something. (for me, mostly)
        print(f"Finished counting, {len(mers)} {ksize}-mers found.")

    return mers  # Sort the dictionary by key alphabetically, return


def tokenize(
    list_of_seqs: list,
    token_size: int,
    goal_redux: float = 0.4,
    sample_size: float = 0.1,
    verbose: bool = True,
):
    # Select a random sample from the input sequences
    samples = sample(list_of_seqs, round(len(list_of_seqs) * sample_size))
    initial_length = sum([len(sample) for sample in samples])
    reduced_length = initial_length

    tokens_to_be_replaced = []
    token_index = 65

    while (1 - reduced_length / initial_length) < goal_redux:
        # For each token in the list, count it in the string. Give us the highest count key.

        # Count the tokens here as though they were kmers
        counts = ray.get(
            [
                count_k_mers.remote(
                    seq, token_size, verbose=False, reverse_compliment_only=False
                )
                for seq in samples
            ]
        )

        # Add all the counters together, and extract the most common k-mer
        counts = reduce(lambda x, y: x + y, counts)
        token = counts.most_common(1)[0][0]

        # And store it
        tokens_to_be_replaced.append(token)

        # Replace the token in the combined samples
        samples = [seq.replace(token, chr(token_index)) for seq in samples]

        # Calculate the percent reduced
        reduced_length = sum([len(sample) for sample in samples])

        if (
            verbose
        ):  # Let the nervous observer know it's doing something. (for me, mostly)
            print(
                f"Replaced {token} with {chr(token_index)} for a {(1-reduced_length/initial_length)*100:.5f}% cumulative savings."
            )

        token_index += 1  # move on to the next token

    tokens_to_be_replaced = [
        (token, chr(index + 65)) for index, token in enumerate(tokens_to_be_replaced)
    ]

    @ray.remote
    def replace_all(seq: str, tokens: list):
        for token, char in tokens_to_be_replaced:
            seq = seq.replace(token, char)
        return seq

    # only return the list of sequences, now tokenized!
    return ray.get(
        [replace_all.remote(seq, tokens_to_be_replaced) for seq in list_of_seqs]
    )


def col_compress(matrix_in: coo_matrix, indices: bool = False) -> csr_matrix:
    matrix = np.sort(matrix_in.toarray(), axis=1)
    indices_out = []
    for col_index in range(matrix.shape[1] - 1):
        if np.array_equal(matrix[:, col_index], matrix[:, col_index + 1]):
            indices_out.append(col_index)
    if indices:
        return indices_out
    return csr_matrix(
        matrix[:, list(set(range(matrix.shape[1])) - set(indices_out))], dtype=np.uint8
    )


@ray.remote
def dict_to_coo_lists(
    isolate: Counter, index: int, key: np.array
) -> Tuple[np.array, np.array, np.array]:
    """
    Turns a dictionary into the parts necessary to make a scipy COO sparse array.
    Requires an input tuple of index and dictionary and a key, so it can be mapped.

    The assumption here is that each kmer count dict will not have all of the kmers present
    in the data, hence the need for a key. This makes the counting a lot faster.
    """
    row_cols = []
    row_data = []

    # for each key, put it in the row/data lists
    for i, kmer, in enumerate(key):  # Loop over the list of found kmers
        value = isolate.pop(
            kmer, 0
        )  # kmer is 'aacgt' or something, remove it from the isolate if present, else get 0
        if value:  # if the value is nonzero, add it to the two lists
            row_data.append(value)  # row_data stores our data
            row_cols.append(i)  # row_cols stores our column indices.

    return (  # return the three indices required. The are: rows, columns, data. rows is just the row index repeated.
        np.full(len(row_cols), index, dtype="u4"),  # array full of kmer index
        np.array(row_cols, dtype="u2"),
        np.array(row_data, dtype="u1"),
    )


def load_data():
    # Assemble the list of files
    resistant_files = [
        "Neisseria/azithromycin/Resistant/" + file
        for file in listdir("Neisseria/azithromycin/Resistant")
        if "fna" in file
    ]

    # get suceptible files, filtering out the .feature files
    susceptible_files = [
        "Neisseria/azithromycin/Susceptible/" + file
        for file in listdir("Neisseria/azithromycin/Susceptible")
        if "fna" in file
    ]

    return (
        [  # SeqIO does the work for us here, each file is treated as one sample. All contigs concatenated.
            "".join([str(contig.seq) for contig in SeqIO.parse(isolate, "fasta")])
            for isolate in resistant_files + susceptible_files
        ],
        np.concatenate(  # labels are zero if resistant, one if susceptible.
            (np.zeros(len(resistant_files)), np.ones(len(susceptible_files)))
        ),
    )


def main():
    token_size = int(argv[1])
    compression_goal = float(argv[2])

    f = open(f"new_log-{token_size}-{compression_goal}.txt", "w")

    ray.init(include_webui=False, include_java=False)

    seqs, y = load_data()

    if token_size > 1:
        seqs = tokenize(seqs, token_size, goal_redux=compression_goal)

    for ksize in range(8, 15):
        start = datetime.now()

        all_kmers = ray.get(
            [count_k_mers.remote(seq, ksize, verbose=False) for seq in seqs]
        )
        print(f"Counting complete for {token_size}, {ksize}")

        ray.put(all_kmers)

        # Create the key from the set of dictionary keys
        key = chain.from_iterable([isolate.keys() for isolate in all_kmers])
        # Sort it to be faster, get rid of the extras
        key = np.array(sorted(set(key)), dtype=f"U{ksize}")

        # Use the pool to convert our dictionaries
        ray.put(key)
        all_kmers = ray.get(
            [dict_to_coo_lists.remote(kmer_dict, index, key) for index, kmer_dict in enumerate(all_kmers)]
        )
        print(f"Conversion complete for {token_size}, {ksize}")

        # The idea here is that we use a COO matrix. Each of these three stores info for each point.
        # At the end, we make one big matrix of shape len(seq), len(key), and feed it out points.
        # example - rows[0] = 5, cols[0] = 3, data[0] = 9
        # So, for the first element we stored, it is the point (5, 3) with value 9.

        # Sort the converted kmers by row index
        all_kmers = sorted(all_kmers, key=lambda x: x[0][0])

        # Initialize the arrays
        rows = np.array([], dtype="u2")
        cols = np.array([], dtype="u4")
        data = np.array([], dtype="u1")

        # Updated out for each isolate
        while all_kmers:
            row, row_cols, row_data = all_kmers.pop(0)

            rows = np.concatenate([rows, row])
            cols = np.concatenate([cols, row_cols])
            data = np.concatenate([data, row_data])

        del all_kmers

        X = coo_matrix((data, (rows, cols)), shape=(len(seqs), len(key)), dtype="u1")

        print(X.dtype)
        # remove this stuff now that we're done with it
        for x in [rows, cols, data]:
            del x

        og_cols = X.shape[1]
        # Compress it, columnwise
        X = col_compress(X)
        after_cols = X.shape[1]
        print(f"Dropped {og_cols - after_cols} columns, {og_cols}, {after_cols}")

        print(f"Fitting for {token_size}, {ksize}")
        clf = RandomForestClassifier(
            n_jobs=32,
            random_state=42,
            n_estimators=512,
            class_weight="balanced",
            oob_score=True,
        )
        # Do ten fold cv
        scores = cross_val_score(clf, X.toarray(), y, cv=10)

        print(  # print the scores and stuff, then write the scores.
            f"{token_size}, {ksize}, {og_cols}, {after_cols}, {scores.mean()},",
            f"{scores.std()}, {scores}, {start}, {datetime.now()}",
        )
        f.write(
            "".join(
                [
                    f"{token_size}, {ksize}, {og_cols}, {after_cols}, {scores.mean()},",
                    f" {scores.std()}, {scores}, {start}, {datetime.now()}\n",
                ]
            )
        )

    f.close()


if __name__ == "__main__":
    main()
