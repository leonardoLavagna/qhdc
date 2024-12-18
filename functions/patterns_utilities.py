#------------------------------------------------------------------------------
# patterns_utilities.py
#
# This module provides utility functions for working with binary patterns
# and similarity measures.
#
# The module includes the following functions:
# - generate_fixed_length_patterns(n,l): Generate a list of binary strings of a fixed length.
# - generate_patterns(n, m): Generates a list of m random bit strings of length n.
# - get_pattern(n, s): Retrieves or converts a bit string pattern of length n.
# - similarity(patterns, search, type="Hamming"): Computes the similarity between 
#   a list of binary patterns and a search pattern using a specified measure.
# - generate_expression(patterns, search): Finds the pattern most similar to the 
#   search pattern and returns it.
# - compress(binary_string, k): Compress a binary string to a fixed length k using SHA-256 hashing.
# - retrieve_original_from_compressed(compressed_string): Retrieve the original binary string from the compressed string using a lookup table.
# - find_keys_by_value(d, target_value): Find all keys in a dictionary that have a specific target value.
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


from typing import List, Tuple, Dict
import random
import hashlib
from scipy.spatial import distance


def generate_fixed_length_patterns(n: int, l: int) -> List[str]:
    """
    Generate a list of binary strings of a fixed length.

    Args:
        n (int): Number of binary strings to generate.
        l (int): The length of each binary string.

    Returns:
        List[str]: List of binary strings of the specified length.
    """
    assert n >= 0 and n<= 2 ** l, "Invalid parameters."
    return [bin(i)[2:].zfill(l) for i in range(n)]
    

def generate_patterns(n: int, m: int) -> List[str]:
    """
    Generate a list of m random bit strings of length n.

    Args:
        n (int): Bit strings length.
        m (int): Number of bit string patterns.

    Returns:
        List[str]: Generated patterns.
    """
    assert n >= 0 and m <= 2 ** n, "Invalid parameters."
    # Dictionary of patterns
    patterns = set()
    while len(patterns) < m:
        pattern = ''.join(str(random.randint(0, 1)) for _ in range(n))
        patterns.add(pattern)
    return list(patterns)



def similarity(patterns: List[str], search: str, type="Hamming") -> Tuple[int, float]:
    """
    Compute the similarity between a list of patterns and a search pattern using the specified similarity measure.

    Args:
        patterns (List[str]): List of binary strings to compare.
        search (str): Binary string to search for within patterns.
        type (str): Type of similarity measure to use. Default is "Hamming".

    Returns:
        tuple: Index of the most similar pattern and the similarity score.

    Raises:
        NotImplementedError: If a similarity measure other than "Hamming" is requested.
    """
    similarities = []
    if type == "Hamming":
        for p in patterns:
            similarities.append(distance.hamming(list(p), list(search)))
        return similarities.index(min(similarities)), min(similarities)
    else:
        raise NotImplementedError


def generate_expression(patterns: List[str], search: str) -> str:
    """
    Generate an expression by finding the pattern most similar to the search pattern.

    Args:
        patterns (List[str]): List of binary strings to compare.
        search (str): Binary string to search for within patterns.

    Returns:
        str: The pattern from the list that is most similar to the search pattern.
    """
    i,_ = similarity(patterns, search)
    return list(patterns)[i]


def compress(binary_string: str, k: int) -> str:
    """
    Compress a binary string to a fixed length k using SHA-256 hashing.

    Args:
        binary_string (str): The binary string to be compressed.
        k (int): The length of the compressed binary string.

    Returns:
        str: The compressed binary string of length k.

    Raises:
        AssertionError: If the length of the binary string is not a multiple of 8.
    """
    assert len(binary_string) % 8 == 0, "Binary string length must be a multiple of 8"
    byte_length = len(binary_string) // 8
    byte_data = int(binary_string, 2).to_bytes(byte_length, byteorder='big')
    hash_object = hashlib.sha256(byte_data)
    hash_digest = hash_object.digest()
    hash_binary_string = ''.join(format(byte, '08b') for byte in hash_digest)
    compressed_binary_string = hash_binary_string[:k]
    return compressed_binary_string


def retrieve_original_from_compressed(compressed_string: str, lookup_table: dict) -> str:
    """
    Retrieve the original binary string from the compressed string using a lookup table.

    Args:
        compressed_string (str): The compressed binary string.
        lookup_table (dict): Lookup table containing previously known strings
    Returns:
        str: The original binary string, or None if not found.
    """
    return lookup_table.get(compressed_string, None)


def find_keys_by_value(d: Dict, target_value) -> List:
    """
    Find all keys in a dictionary that have a specific target value.

    Args:
        d (Dict): The dictionary to search.
        target_value: The value to search for.

    Returns:
        List: A list of keys that have the target value.
    """
    keys = [key for key, value in d.items() if value == target_value]
    return keys