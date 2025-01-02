import gzip


def compression_entropy(sequence):
    # Convert the sequence to bytes
    sequence_bytes = bytes(sequence)

    # Compress the sequence
    compressed_sequence = gzip.compress(sequence_bytes)

    # Calculate entropy based on the compression ratio
    entropy = len(compressed_sequence) / len(sequence_bytes)
    return entropy
