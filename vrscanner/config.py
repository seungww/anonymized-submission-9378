import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="VRScanner: VR Traffic Fingerprinting Attack")

    parser.add_argument("--path", type=str, nargs="+", required=True,
                        help="Path(s) to the dataset CSV file(s). Supports multiple input files.")
    parser.add_argument("--unknown_path", type=str, nargs="+", default=None,
                        help="Path(s) to the unknown CSV file(s) for evaluation of app-launch-detection.")
    parser.add_argument("--pktcount", type=int, default=500,
                        help="Number of packets to include in each sequence window (default: 100)")
    parser.add_argument("--kfold", type=int, default=1,
                        help="Number of folds for k-fold cross-validation (default: 1 for no cross-validation)")
    parser.add_argument("--model", type=str.lower, default=['gru'], nargs="+",
                        choices=["rnn", "lstm", "gru", "birnn", "bilstm", "bigru", "cnn", "transformer"],
                        help="Model architecture to use. Options: RNN/LSTM/GRU, biRNN/biLSTM/biGRU, CNN, Transformer")
    parser.add_argument("--norm", type=str.lower, default=['token'], nargs="+",
                        choices=["none", "minmax", "zscore", "token", "binary", "maxabs", "l1norm", "l2norm", "power", "quantile", "robust"],
                        help="Normalization strategy to apply per input feature.")
    parser.add_argument("--input_dim", type=int, default=512, help="Input feature dimension")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of RNN layers")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of stacked RNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout between RNN layers")
    parser.add_argument("--fusion_dim", type=int, default=256, help="Fusion layer size")
    parser.add_argument("--fc_hidden_size", type=int, default=128, help="Hidden layer size of FC head")
    parser.add_argument("--cnn_kernel_size", type=int, default=3, help="Kernel size for 1D-CNN")
    parser.add_argument("--cnn_layers", type=int, default=2, help="Number of Conv1d layers for 1D-CNN")
    parser.add_argument("--transformer_heads", type=int, default=4, help="Number of attention heads for Transformer")
    parser.add_argument("--transformer_ff", type=int, default=512, help="Feedforward dimension for Transformer")
    parser.add_argument("--transformer_layers", type=int, default=2, help="Number of Transformer encoder layers")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout for Transformer encoder")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    parser.add_argument("--feature_attention", action="store_true", help="Enable feature-level attention score")
    parser.add_argument("--timestep_attention", action="store_true", help="Enable timestep-level attention score")
    parser.add_argument("--strict", action="store_true", help="Enable retraining in bad local minimum")
    parser.add_argument("--window_size", type=int, default=300, help="Window size for sliding window")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for sliding window")

    parser.add_argument("--debug_path", type=str, default="output.debug", help="Debug log file path")
    parser.add_argument("--leaderboard_path", type=str, default="leaderboard.csv", help="Leaderboard output path")
    parser.add_argument("--step1", action="store_true", help="Enable step1 mode")
    parser.add_argument("--step2", action="store_true", help="Enable step2 mode")
    parser.add_argument("--step3", action="store_true", help="Enable step3 mode")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--sliding_window_evaluation", action="store_true", help="Enable trainging mode with sliding window")
    parser.add_argument("--longitudinal_evaluation", action="store_true", help="Enable longitudinal evaluation")
    parser.add_argument("--openworld_evaluation", action="store_true", help="Enable open-world evaluation")
    parser.add_argument("--app_launch_detection_evaluation", action="store_true",
                        help="Enable app launch detection evaluation")

    return parser.parse_args()
