from typing import Dict, Iterable, Callable


def parse_regularizer(regularizer_argument: str = "") -> Dict[str, float]:
    regularizers = {}

    regularizer_arguments_list = regularizer_argument.split("_")

    if len(regularizer_arguments_list) % 2 == 1:
        raise ValueError("Each regularizer should have only one regularizer decay")

    for i in range(0, len(regularizer_arguments_list), 2):
        regularizers[regularizer_arguments_list[i]] = float(regularizer_arguments_list[i+1])

    return regularizers
