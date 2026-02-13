import argparse
from camel.camel_model import CAMEL

def main():
    parser = argparse.ArgumentParser(description="CAMEL")
    parser.add_argument("--mode", type=str, choices=['forecast', 'base', 'ecgbench'], default='base')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--ecgs", type=str, default=None, nargs='+')
    parser.add_argument("--ecg-configs", type=str, default=None, nargs='+')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=64,
        help="Top-k sampling cutoff (set <=0 to disable).",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling cumulative probability cutoff.",
    )
    parser.add_argument(
        "--min-p",
        dest="min_p",
        type=float,
        default=0.0,
        help="Minimum per-token probability threshold applied after temperature scaling.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per sample.",
    )
    args = parser.parse_args()
    
    model = CAMEL(mode=args.mode, device=args.device)
    output, prompt = model.run(args)
    
    print(f'Prompt: {prompt}')
    print(f'Prediction: {output}')

if __name__ == "__main__":
    main()
