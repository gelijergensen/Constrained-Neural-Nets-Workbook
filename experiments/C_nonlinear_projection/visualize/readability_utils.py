"""Tools to help with making plots easier to read"""


def _correct_and_clean_labels(labels):
    return [_clean_label(_correct_label(label)) for label in labels]


def _correct_label(label):
    return label if label != "step_optimizer" else "backward_pass"


def _clean_label(label):
    """Replaces all underscores with spaces and capitalizes first letter"""
    return label.replace("_", " ").capitalize() if label is not None else None
