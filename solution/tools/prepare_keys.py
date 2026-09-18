"""Discover keys from real noisy inputs; never invent a competition key list."""
import argparse
from pathlib import Path


def discover(root, expected_count):
    root = Path(root).expanduser().resolve()
    base = root / 'shapenet' if (root / 'shapenet').is_dir() else root
    keys = sorted('shapenet/' + p.parent.relative_to(base).as_posix()
                  for p in base.glob('*/*/noisy.npy') if p.is_file())
    if len(keys) != expected_count or len(set(keys)) != expected_count:
        raise ValueError(f'Expected {expected_count} unique noisy inputs, found {len(keys)} in {base}')
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--expected-count', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'test_200_keys.txt')
    args = parser.parse_args()
    keys = discover(args.input_root, args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(keys) + '\n', encoding='utf8')
    print(f'Wrote {len(keys)} observed keys to {args.output}')


if __name__ == '__main__':
    main()
