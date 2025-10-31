import torch
from vrscanner.config import parse_arguments
from vrscanner.utils import init_logging, init_leaderboard
from vrscanner.experiment.general import (
    process_step1,
    process_step2,
    process_step3,
    process_train,
    sliding_window_evaluation,
)
from vrscanner.experiment.longitudinal import longitudinal_evaluation
from vrscanner.experiment.openworld import openworld_evaluation


def main():
    args = parse_arguments()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_logging(args.debug_path)
    init_leaderboard(args.leaderboard_path)

    if args.step1:
        process_step1(args)

    if args.step2:
        process_step2(args)

    if args.step3:
        process_step3(args)

    if args.train:
        process_train(args)

    if args.sliding_window_evaluation:
        sliding_window_evaluation(args)

    if args.longitudinal_evaluation:
        longitudinal_evaluation(args)

    if args.openworld_evaluation:
        openworld_evaluation(args)


if __name__ == "__main__":
    main()

