
import numpy as np
from beans_zero.evaluate import *
from beans_zero.post_processor import *
from Levenshtein import distance as levenshtein_distance

def macro_f1_baselines(class_counts):
    """
    Given class counts, returns:
    (macro_f1_majority, macro_f1_uniform_random)
    """
    counts = np.array(class_counts, dtype=float)
    N = counts.sum()
    K = len(counts)

    # --- Majority-class model ---
    majority = np.argmax(counts)
    F1s_majority = []
    for i in range(K):
        if i == majority:
            TP = counts[i]
            FP = N - counts[i]
            FN = 0
        else:
            TP = 0
            FP = 0
            FN = counts[i]
        denom = 2*TP + FP + FN
        F1s_majority.append(0 if denom == 0 else 2*TP/denom)
    macro_f1_majority = np.mean(F1s_majority)

    # --- Uniform random model ---
    F1s_random = []
    for i in range(K):
        TP = counts[i]/K
        FP = (N - counts[i])/K
        FN = counts[i]*(1 - 1/K)
        denom = 2*TP + FP + FN
        F1s_random.append(0 if denom == 0 else 2*TP/denom)
    macro_f1_random = np.mean(F1s_random)

    return macro_f1_majority, macro_f1_random

def accuracy_baseline(class_counts):
    counts = np.array(class_counts, dtype=float)
    N = counts.sum()
    majority = np.argmax(counts)
    accuracy_majority = counts[majority] / N
    accuracy_random = 1 / len(counts)
    return accuracy_majority, accuracy_random

def get_nearest_label(self, text: str) -> str:
    """
    Find the nearest label to the text using levenshtein distance.
    If the distance is greater than max_distance_for_match, return 'None'.

    If multi_label is True, return a comma-separated string of labels.

    Arguments
    ---------
    text : str
        The text to match

    Returns
    -------
    str
        The matched label

    Examples
    --------
    >>> processor = EvalPostProcessor(set(["dog", "cat", "bird"]),"classification")
    >>> processor.get_nearest_label("dog")
    'dog'
    """
    # Check for exact match first
    text = self.remove_eos_token(text)
    if text in self.target_label_set:
        return text

    nearest_label = min(
        list(self.target_label_set),
        key=lambda label: levenshtein_distance(text, label),
    )
    if (
        levenshtein_distance(nearest_label, text) > self.max_levenstein_distance
        #and self.multi_label
    ):
        return "None"  # DETECTION only: no strong match, choose None

    return nearest_label

EvalPostProcessor.get_nearest_label = get_nearest_label