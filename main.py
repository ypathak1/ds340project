"""Entry point to run the webcam demo or data collector.

Usage:
  python main.py demo        # run the live demo (inference)
  python main.py collect     # run the mouth-region data collector

The script delegates to `src.pipeline` and `src.data_collector`.
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['demo', 'collect'], help='Which mode to run')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='data/collected')
    parser.add_argument('--classes', default='neutral,frown,smile,tongue,open')
    parser.add_argument('--smoother', type=int, default=7)
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    # Ensure project root is on sys.path so src can be imported when run as script
    # (Useful when running python main.py)
    if '' not in sys.path:
        sys.path.insert(0, '')

    if args.mode == 'demo':
        from src.pipeline import run_demo
        run_demo(device=args.device, model_checkpoint=args.checkpoint, smoother_win=args.smoother)
    elif args.mode == 'collect':
        from src.data_collector import run_collector
        classes = [c.strip() for c in args.classes.split(',') if c.strip()]
        run_collector(output_dir=args.out, classes=classes)


if __name__ == '__main__':
    main()
