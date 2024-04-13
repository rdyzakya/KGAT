import torch

KG_MASK = "<KG>"
SUBJECT_MASK = "<SUBJECT>"
RELATION_MASK = "<RELATION>"
OBJECT_MASK = "<OBJECT>"

def apply_template(text, subject, relation, objects=None):
    text = text.replace(SUBJECT_MASK, subject).replace(RELATION_MASK, relation)
    if objects is None:
        return text
    return text.replace(OBJECT_MASK, str(objects))

def inverse_sigmoid(y):
    """
    Compute the inverse of the sigmoid function.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the inverse sigmoid function.
    """
    # Clip the input tensor to avoid numerical instability
    y = torch.clamp(y, min=1e-7, max=1.0 - 1e-7)

    # Apply the inverse sigmoid function
    x = torch.log(y / (1 - y))

    return x