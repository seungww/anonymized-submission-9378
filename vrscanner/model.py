import math
import torch
import torch.nn as nn


class VRScannerModel(nn.Module):
    def __init__(
        self,
        model_types,
        output_size,
        input_layers,
        input_sizes,
        input_dims,
        hidden_size,
        num_layers,
        dropout,
        fusion_dim=256,
        fc_hidden_size=128,
        cnn_kernel_size=3,
        cnn_layers=2,
        transformer_heads=4,
        transformer_ff=512,
        transformer_layers=2,
        transformer_dropout=0.1,
    ):
        super(VRScannerModel, self).__init__()

        self.input_modules = nn.ModuleList()
        self.sequence_encoders = nn.ModuleList()
        self.transformer_input_proj = nn.ModuleList()
        self.attn_nets = nn.ModuleList()          # timestep-level attention
        self.projection_layers = nn.ModuleList()

        self.input_specs = []
        self.encoder_types = []

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

            if model_type in rnn_dict:
                rnn_class = rnn_dict[model_type]
                self.sequence_encoders.append(
                    rnn_class(rnn_input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout,
                              bidirectional=bidirectional)
                )
                rnn_output_dim = hidden_size * (2 if bidirectional else 1)
                self.transformer_input_proj.append(nn.Identity())
                self.encoder_types.append("rnn")
            elif model_type == "cnn":
                if cnn_kernel_size < 1:
                    raise ValueError("cnn_kernel_size must be >= 1")
                padding = cnn_kernel_size // 2
                layers = []
                in_channels = rnn_input_size
                for _ in range(max(1, cnn_layers)):
                    layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=cnn_kernel_size, padding=padding))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(p=dropout))
                    in_channels = hidden_size
                self.sequence_encoders.append(nn.Sequential(*layers))
                rnn_output_dim = hidden_size
                self.transformer_input_proj.append(nn.Identity())
                self.encoder_types.append("cnn")
            elif model_type == "transformer":
                if hidden_size % transformer_heads != 0:
                    raise ValueError("hidden_size must be divisible by transformer_heads")
                if rnn_input_size == hidden_size:
                    self.transformer_input_proj.append(nn.Identity())
                else:
                    self.transformer_input_proj.append(nn.Linear(rnn_input_size, hidden_size))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=transformer_heads,
                    dim_feedforward=transformer_ff,
                    dropout=transformer_dropout,
                    batch_first=True,
                )
                self.sequence_encoders.append(nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers))
                rnn_output_dim = hidden_size
                self.encoder_types.append("transformer")
            else:
                raise ValueError(f"Unsupported model type: {model_type_raw}")

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

        for i, (input_module, encoder, encoder_type, attn_net, proj_layer) in enumerate(
                zip(self.input_modules, self.sequence_encoders, self.encoder_types, self.attn_nets, self.projection_layers)):
            input_x = x[i]  # shape: [B, T] or [B, T, D]
            embedded = input_module(input_x)  # [B, T, D]

            if encoder_type == "rnn":
                out, _ = encoder(embedded)      # [B, T, H]
            elif encoder_type == "cnn":
                conv_in = embedded.transpose(1, 2)  # [B, D, T]
                conv_out = encoder(conv_in)         # [B, H, T]
                out = conv_out.transpose(1, 2)      # [B, T, H]
            elif encoder_type == "transformer":
                proj = self.transformer_input_proj[i](embedded)
                proj = proj + self._positional_encoding(proj.size(1), proj.size(2), proj.device)
                out = encoder(proj)                 # [B, T, H]
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

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

    @staticmethod
    def _positional_encoding(seq_len, dim, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
