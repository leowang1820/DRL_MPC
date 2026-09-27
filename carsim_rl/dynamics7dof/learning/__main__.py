"""First-layer residual framework. No subcommand starts a CarSim simulation."""

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build', help='Create experimental derivative residual dataset')
    build.add_argument('--train', action='append', required=True)
    build.add_argument('--validation', action='append', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--stride', type=int, default=5)
    train = sub.add_parser('train', help='Train bootstrapped independent residual networks')
    train.add_argument('--dataset', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--hidden', type=int, default=64)
    train.add_argument('--members', type=int, default=5)
    train.add_argument('--device', default='cpu')
    train.add_argument('--seed', type=int, default=42)
    for name in ('evaluate', 'shadow'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--checkpoint', required=True)
        cmd.add_argument('--capture', required=True)
        cmd.add_argument('--output', required=True)
        if name == 'shadow':
            cmd.add_argument('--adapt', action='store_true', help='Enable bounded candidate updates during chronological replay')
            cmd.add_argument('--update-every', type=int, default=200)
    args = p.parse_args(argv)
    if args.command == 'build':
        from .data import build_dataset
        build_dataset(args.train, args.validation, args.output, args.stride)
    else:
        import torch
        # Small networks in this workflow; avoid excessive BLAS thread overhead.
        torch.set_num_threads(1)
        if args.command == 'train':
            from .ensemble import train_dataset
            train_dataset(args.dataset, args.output, epochs=args.epochs, hidden=args.hidden,
                          count=args.members, device=args.device, seed=args.seed)
        elif args.command == 'evaluate':
            from .workflow import evaluate_checkpoint
            evaluate_checkpoint(args.checkpoint, args.capture, args.output)
        else:
            from .workflow import shadow_replay
            shadow_replay(args.checkpoint, args.capture, args.output, adapt=args.adapt, update_every=args.update_every)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
