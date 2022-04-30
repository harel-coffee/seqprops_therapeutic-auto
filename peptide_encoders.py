#####################################################################################
##  This script contains helpers for other three representation schemes:           ##
##      -- peptide properties                                                      ##
##      -- one-hot encoding                                                        ## 
##      -- word embedding                                                          ##
#####################################################################################

import numpy as np
import peptides

class PeptidePropertiesEncoder:
    def encode(self, sequences):
        groups = (
            ('A', 'C', 'G', 'S', 'T'),                                  # Tiny
            ('A', 'C', 'D', 'G', 'N', 'P', 'S', 'T', 'V'),              # Small 
            ('A', 'I', 'L', 'V'),                                       # Aliphatic
            ('F', 'H', 'W', 'Y'),                                       # Aromatic
            ('A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y'),    # Non-polar
            ('D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T'),              # Polar
            ('D', 'E', 'H', 'K', 'R'),                                  # Charged
            ('H', 'K', 'R'),                                            # Basic
            ('D', 'E')                                                  # Acidic
        )

        X = []
        for sequence in sequences:
            sequence = sequence.upper()

            peptide = peptides.Peptide(sequence)
            x = [
                peptide.cruciani_properties()[0],
                peptide.cruciani_properties()[1],
                peptide.cruciani_properties()[2],
                peptide.instability_index(),
                peptide.boman(),
                peptide.hydrophobicity("Eisenberg"),
                peptide.hydrophobic_moment(angle=100, window=min(len(sequence), 11)),
                peptide.aliphatic_index(),
                peptide.isoelectric_point("Lehninger"),
                peptide.charge(pH=7.4, pKscale="Lehninger"),
            ]

            # Count tiny, small, aliphatic, ..., basic and acidic amino acids
            for group in groups:
                count = 0
                for amino in group:
                    count += sequence.count(amino)
                x.append(count)
                x.append(count / len(sequence))
            X.append(x)
        return np.array(X)


class IntegerTokensEncoder:
    def __init__(self, max_seq_len=None, stop_signal=True):
        self.max_seq_len = max_seq_len
        self.stop_signal = stop_signal

    def encode(self, sequences):
        vocab = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
            }
        encoded_sequences = []
        max_seq_len = 0
        for sequence in sequences:
            sequence = sequence.upper()
            encoded_sequence = []
            for amino_acid in sequence:
                if amino_acid not in vocab:
                    raise ValueError("Unknown amino acid \"{}\"".format(amino_acid))
                else:
                    encoded_sequence.append(vocab[amino_acid])
            encoded_sequences.append(encoded_sequence)
            max_seq_len = max(max_seq_len, len(sequence))

        if self.max_seq_len is not None:
            max_seq_len = self.max_seq_len
        max_seq_len += 1

        if self.stop_signal:
            for sequence in encoded_sequences:
                while len(sequence) < max_seq_len:
                    sequence.append(len(vocab))
        
        return np.array(encoded_sequences)

class OneHotEncoder:
    def __init__(self, max_seq_len=None, stop_signal=True):
        self.max_seq_len = max_seq_len
        self.stop_signal = stop_signal

    def encode(self, sequences):
        vocab = {
            'A': 0, 'R': 1, 'N':2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
            }

        if self.stop_signal:
            vec_length = 21
        else:
            vec_length = 20
        encoded_sequences = []
        max_seq_len = 0
        for sequence in sequences:
            sequence = sequence.upper()
            encoded_sequence = []
            for amino_acid in sequence:
                vec = [0 for _ in range(vec_length)]
                pos = vocab[amino_acid]
                vec[pos] = 1
                encoded_sequence.append(vec)
            encoded_sequences.append(encoded_sequence)
            max_seq_len = max(max_seq_len, len(sequence))
        
        if self.max_seq_len is not None:
            max_seq_len = self.max_seq_len
        max_seq_len += 1

        if self.stop_signal:
            for sequence in encoded_sequences:
                while len(sequence) < max_seq_len:
                    vec = [0 for _ in range(vec_length)]
                    vec[-1] = 1
                    sequence.append(vec)
        return np.array(encoded_sequences)

