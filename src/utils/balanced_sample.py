####################################################
###
###                     FUNCTION
###
####################################################


def balanced_sample(group, subset_percentage):
    """
    Perform balanced sampling on a DataFrame group.

    Args:
        group (DataFrame): The DataFrame or group to sample from.
        subset_percentage (float): The fraction of the group to sample (between 0.0 and 1.0).

    Returns:
        DataFrame: A randomly sampled fraction of the group, based on the given percentage.
    """
    return group.sample(frac=subset_percentage)
