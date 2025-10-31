import torch
from vrscanner.model import VRScannerModel


def get_model(args, num_of_classes, num_tokens_list):
    models = args.model
    norms = args.norm
    input_dim = args.input_dim
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    fusion_dim = args.fusion_dim
    fc_hidden_size = args.fc_hidden_size

    output_size = num_of_classes
    input_layers = []
    input_sizes = []
    input_dims = []

    for norm, num_tokens in zip(norms, num_tokens_list):
        if norm in ["minmax", "zscore", 'binary', 'maxabs', 'l1norm', 'l2norm', 'power', 'quantile', 'robust']:
            input_layers.append("linear")
            input_sizes.append(1)
            input_dims.append(input_dim)
        elif norm == "token":
            input_layers.append("embedding")
            input_sizes.append(num_tokens)
            input_dims.append(input_dim)
        else:
            input_layers.append("identity")
            input_sizes.append(1)
            input_dims.append(input_dim)

    return VRScannerModel(
        models,
        output_size=output_size,
        input_layers=input_layers,
        input_sizes=input_sizes,
        input_dims=input_dims,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fusion_dim=fusion_dim,
        fc_hidden_size=fc_hidden_size
    )
