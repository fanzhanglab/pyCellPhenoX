####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd

####################################################
###
###                     FUNCTION
###
####################################################


def check_indices(a, b):
    """Check that the indices are matching for the dataframes and assign indices if they aren't.

    Args:
        a (pd.DataFrame): DataFrame 1
        b (pd.DataFrame): DataFrame 2

    Returns:
        a, b (pd.DataFrame, pd.DataFrame):
    """

    # We assume here that we prefer the indices that are strings (i.e., cell names/barcodes)
    if a.index.dtype == "object" and b.index.dtype != "object":
        print("Synchronizing indices: a has string indices.")
        b.index = a.index
    elif b.index.dtype == "object" and a.index.dtype != "object":
        print("Synchronizing indices: b has string indices.")
        a.index = b.index
    else:
        print(
            "Both DataFrames already have matching index types or no string indices detected. Leaving alone."
        )

    return a, b
