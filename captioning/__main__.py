import argparse
import json
from .pipeline import run, predict

parser = argparse.ArgumentParser(description='Restore the original Flickr8k captioning experiment without implicit downloads')
parser.add_argument('command', choices=['smoke', 'train', 'predict'])
parser.add_argument('--encoder', choices=['mobilenet', 'inception'], default='mobilenet')
parser.add_argument('--variant', choices=['lstm', 'blstm'], default='lstm')
parser.add_argument('--output', default='artifacts/smoke')
parser.add_argument('--data-root')
parser.add_argument('--encoder-weights')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--artifact-dir')
parser.add_argument('--image')
args = parser.parse_args()
try:
    if args.command == 'predict':
        if not all([args.artifact_dir, args.image, args.encoder_weights]):
            parser.error('predict requires --artifact-dir, --image and --encoder-weights')
        print(predict(args.artifact_dir, args.image, args.encoder_weights))
    else:
        print(json.dumps(run(args.output, args.encoder, args.variant, args.command == 'smoke',
                             args.data_root, args.encoder_weights, args.epochs, args.batch_size, args.seed), indent=2))
except (ValueError, FileNotFoundError) as error:
    parser.exit(2, f'{error}\n')
