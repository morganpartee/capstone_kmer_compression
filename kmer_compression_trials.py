from Bio import SeqIO
from os import listdir
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from functools import partial
from itertools import chain, product
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from multiprocessing import Pool
from datetime import datetime


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


def count_k_mers(seq: Tuple[int, str], ksize: int, verbose=False):
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
    index, seq = seq  # unpack the sequence, hold index for the return

    mers = Counter()
    for i in range(len(seq) - ksize):
        mer = seq[i : i + ksize]  # Each k-mer is a sliding window piece of the sequence
        rc = reverse_compliment(mer)
        if lexical_score(mer) < lexical_score(rc):  # update the counter with:
            mers.update([mer])  # the  k-mer if it's lower lexically
        else:
            mers.update([rc])  # Else the reverse compliment
        if (
            verbose
        ):  # Let the nervous observer know it's doing something. (for me, mostly)
            print(f"Finished counting {index}, {len(mers)} {ksize}-mers found.")
    return (
        index,
        dict(sorted(mers.items(), key=lambda x: x[0])),
    )  # Sort the dictionary by key alphabetically, return


def tokenize(list_of_seqs: list, token_size: int, verbose=True, dna_chars="acgt"):
    """
    This function is a naive compression algorithm. It replaces common sequences
    with a token (starting at capital A), replacing them until it no longer saves
    us 1% of the total length.

    This is slow! It must tokenize all of the data at once and each sample independently
    to maintain the tokenization scheme across samples.

    Examples:
    # Just for you rob:
    In [21]: tokenize(["the quick brown dog the lazy fox"], 3, dna_chars='the')
    Out[21]: ['A quick brown dog A lazy fox']

    In [23]: tokenize(["aaacccgggttttcgatcgatcgatcgatcgatatcggctatatacgc"], 3, True)
    Replaced tcg with A for a 0.25000% savings.
    Replaced ata with B for a 0.11111% savings.
    Replaced aaa with C for a 0.06250% savings.
    Replaced acg with D for a 0.06667% savings.
    Replaced ccc with E for a 0.07143% savings.
    Replaced gct with F for a 0.07692% savings.
    Replaced ggg with G for a 0.08333% savings.
    Replaced ttt with H for a 0.09091% savings.
    Replaced aaa with I for a 0.00000% savings.
    Out[23]: ['CEGHAaAaAaAaABAFBtDc']
    """
    seq = "".join(
        list_of_seqs
    )  # Combine everything, so that we can count/tokenize this
    size_redux = 0  # basically a 'true' for the while loop
    token_index = 65  # start with capital A, try chr(65)
    # Since this is DNA represented by acgt, we can assume the most common sequences
    # are only made up of acgt. N is left out as it is uncommon.
    token_list = tuple("".join(toke) for toke in product(dna_chars, repeat=token_size))

    while size_redux < 0.99:
        before = len(seq)
        # For each token in the list, count it in the string. Give us the highest count key.
        token = max(
            [(toke, seq.count(toke)) for toke in token_list], key=lambda x: x[1]
        )[0]

        # Replace the token in the combined samples
        seq = seq.replace(token, chr(token_index))

        # And, replace the token in each contig
        list_of_seqs = [x.replace(token, chr(token_index)) for x in list_of_seqs]

        # Calculate the percent reduced
        size_redux = len(seq) / before

        if (
            verbose
        ):  # Let the nervous observer know it's doing something. (for me, mostly)
            print(
                f"Replaced {token} with {chr(token_index)} for a {1-size_redux:.5f}% savings."
            )

        token_index += 1  # move on to the next token
    return list_of_seqs  # only return the list of sequences, now tokenized!


def dict_to_coo_lists(
    isolate: Tuple[int, dict], key: np.array
) -> Tuple[np.array, np.array, np.array]:
    """
    Turns a dictionary into the parts necessary to make a scipy COO sparse array.
    Requires an input tuple of index and dictionary and a key, so it can be mapped.

    The assumption here is that each kmer count dict will not have all of the kmers present
    in the data, hence the need for a key. This makes the counting a lot faster.
    """
    i, isolate = isolate
    row_cols = []
    row_data = []

    # for each key, put it in the row/data lists
    for index, kmer, in enumerate(key):  # Loop over the list of found kmers
        value = isolate.pop(
            kmer, 0
        )  # kmer is 'aacgt' or something, remove it from the isolate if present, else get 0
        if value:  # if the value is nonzero, add it to the two lists
            row_data.append(value)  # row_data stores our data
            row_cols.append(index)  # row_cols stores our column indices.

    return (  # return the three indices required. The are: rows, columns, data. rows is just the row index repeated.
        np.full(len(row_cols), i, dtype="u4"),  # array full of kmer index
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
    f = open("log.txt", "w")

    start = datetime.now()

    for token_size in [4, 10]:
        with Pool(12) as p:

            seqs, y = load_data()
            seqs = tokenize(seqs, token_size)
            seqs = [(i, seq) for i, seq in enumerate(tokenize(seqs, token_size))]

            for ksize in range(2, 5):

                mappable_count_k_mers = partial(count_k_mers, ksize=ksize)
                all_kmers = p.map(mappable_count_k_mers, seqs)
                print(f"Counting complete for {token_size}, {ksize}")

                # Create the key from the set of dictionary keys
                key = chain.from_iterable([isolate[1].keys() for isolate in all_kmers])
                # Sort it to be faster, get rid of the extras
                key = np.array(sorted(set(key)), dtype=f"U{ksize}")

                # Set the key to key for dict_to_coo_lists
                mappable_dict_to_list = partial(dict_to_coo_lists, key=key)

                # Use the pool to convert our dictionaries
                all_kmers = p.map(mappable_dict_to_list, all_kmers)
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

                X = coo_matrix(
                    (data, (rows, cols)), shape=(len(seqs), len(key)), dtype="u1"
                ).tocsr()

                # clean up before saving, pickle can be memory wasteful
                for x in [rows, cols, data]:
                    del x

                print(f"Fitting for {token_size}, {ksize}")
                clf = RandomForestClassifier(
                    n_jobs=32,
                    random_state=42,
                    n_estimators=2048,
                    class_weight="balanced",
                    oob_score=True,
                )
                # Do ten fold cv
                scores = cross_val_score(clf, X, y, cv=10)

                print(  # print the scores and stuff, then write the scores.
                    f"{token_size}, {ksize}, {len(key)}, {scores.mean()},",
                    f"{scores.std()}, {scores}, {start}, {datetime.now()}",
                )
                f.write(
                    "".join(
                        [
                            f"{token_size}, {ksize}, {len(key)}, {scores.mean()},",
                            f" {scores.std()}, {scores}, {start}, {datetime.now()}",
                        ]
                    )
                )

    f.close()


if __name__ == "__main__":
    main()
