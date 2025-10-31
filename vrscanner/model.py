import torch
import torch.nn as nn


class VRScannerModel(nn.Module):
    def __init__(self, model_types, output_size, input_layers, input_sizes, input_dims,
                 hidden_size, num_layers, dropout, fusion_dim=256, fc_hidden_size=128):
        super(VRScannerModel, self).__init__()

        self.input_modules = nn.ModuleList()
        self.rnn_layers = nn.ModuleList()
        self.attn_nets = nn.ModuleList()          # timestep-level attention
        self.projection_layers = nn.ModuleList()

        self.input_specs = []

        rnn_dict = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU
        }

        self.rnn_output_dims = []

        for i, (in_layer, model_type_raw) in enumerate(zip(input_layers, model_types)):
            model_type = model_type_raw.lower()
            bidirectional = False

            if model_type.startswith("bi"):
                bidirectional = True
                model_type = model_type[2:]

            if model_type not in rnn_dict:
                raise ValueError(f"Unsupported model type: {model_type_raw}")

            rnn_class = rnn_dict[model_type]

            # Input layer
            if in_layer == "embedding":
                self.input_modules.append(nn.Embedding(input_sizes[i], input_dims[i]))
                rnn_input_size = input_dims[i]
            elif in_layer == "linear":
                self.input_modules.append(nn.Linear(1, input_dims[i]))
                rnn_input_size = input_dims[i]
            elif in_layer == "identity":
                self.input_modules.append(nn.Identity())
                rnn_input_size = 1
            else:
                raise ValueError(f"Unknown input layer type: {in_layer}")

            self.rnn_layers.append(
                rnn_class(rnn_input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
            )

            rnn_output_dim = hidden_size * (2 if bidirectional else 1)
            self.rnn_output_dims.append(rnn_output_dim)

            self.attn_nets.append(nn.Sequential(
                nn.Linear(rnn_output_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            ))

            self.projection_layers.append(nn.Linear(rnn_output_dim, fusion_dim))
            self.input_specs.append((in_layer, model_type_raw))

        # Feature-level attention (sample-dependent)
        self.feature_attn_net = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden_size, output_size)
        )

    def forward(self, x, return_attn_weights=False, return_feature_weights=False):
        rnn_outputs = []
        attn_weights_all = []

        for i, (input_module, rnn_layer, attn_net, proj_layer) in enumerate(
                zip(self.input_modules, self.rnn_layers, self.attn_nets, self.projection_layers)):
            input_x = x[i]  # shape: [B, T] or [B, T, D]
            embedded = input_module(input_x)  # [B, T, D]
            out, _ = rnn_layer(embedded)      # [B, T, H]

            # Timestep-level attention
            attn_score = attn_net(out)                 # [B, T, 1]
            attn_weight = torch.softmax(attn_score, dim=1)  # [B, T, 1]
            context = torch.sum(out * attn_weight, dim=1)   # [B, H]

            proj_out = proj_layer(context)                  # [B, fusion_dim]
            rnn_outputs.append(proj_out)

            if return_attn_weights:
                attn_weights_all.append(attn_weight.squeeze(-1).detach())  # [B, T]

        # Stack projected feature outputs → shape: [B, N_feature, fusion_dim]
        feature_vecs = torch.stack(rnn_outputs, dim=1)  # [B, N, D]

        # Feature-level attention (sample-dependent)
        raw_feature_scores = self.feature_attn_net(feature_vecs)  # [B, N, 1]
        feature_weights = torch.softmax(raw_feature_scores, dim=1)  # [B, N, 1]

        # Weighted sum over features
        fused = torch.sum(feature_vecs * feature_weights, dim=1)  # [B, D]

        output = self.fc(fused)

        if return_attn_weights or return_feature_weights:
            results = [output]
            if return_attn_weights:
                results.append(attn_weights_all)  # List of [B, T]
            if return_feature_weights:
                results.append(feature_weights.squeeze(-1).detach())  # [B, N]
            return tuple(results)

        return output

