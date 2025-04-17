import numpy as np
import math
import pandas as pd

def make_cyclic(x, x_max):
    """
    Computes the cyclic representation of a variables.

    Args:
        x (array_like): Input array to be transformed

    Returns:
        (array_like): x axis of transform
        (array_like): y axis of transform
    """
    if x_max is None:
        x_max = x.max()

    x_norm = 2 * math.pi * x / x_max
    return np.sin(x_norm), np.cos(x_norm)


def harmonics(period, nwave, fun=np.sin):
    """
    Generate a sinusoidal or cosinusoidal wave function.
    
    Parameters:
        period (float): The period of the wave.
        nwave (float): The wave frequency (number of waves).
        fun (function): The function to apply, e.g., np.sin or np.cos.
    
    Returns:
        function: A function that takes x and returns the harmonic value.
    """
    def h(x):
        return fun(2 * np.pi * nwave * x / period)
    return h


def generate_harmonics(v, nwaves):
    """
    Generate sine and cosine harmonics for a given vector and wave frequencies.

    Parameters:
        v (array-like): Input vector of values.
        nwaves (array-like): List of wave frequencies to generate harmonics.

    Returns:
        pd.DataFrame: A DataFrame containing the sine and cosine harmonics.
    """
    # Remove NaN or invalid values and find the maximum of v
    v_clean = np.array(v)[~np.isnan(v)]  # Filter out NaN
    mx = np.max(v_clean)

    # Generate sine harmonics
    res_sin = [harmonics(mx, n, np.sin)(v) for n in nwaves]

    # Generate cosine harmonics
    res_cos = [harmonics(mx, n, np.cos)(v) for n in nwaves]

    # Combine results and create names
    column_names = [f"cos_{n}" for n in nwaves] + [f"sin_{n}" for n in nwaves]
    harmonics_matrix = np.column_stack(res_cos + res_sin)

    # Convert to a DataFrame
    df_harmonics = pd.DataFrame(harmonics_matrix, columns=column_names)

    return df_harmonics
