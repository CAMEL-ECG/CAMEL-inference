import argparse
from src.camel.camel_model import CAMEL

def main():
    parser = argparse.ArgumentParser(description="CAMEL")
    parser.add_argument("--mode", type=str, choices=['forecast', 'base', 'ecgbench'], required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--ecg", type=str, required=True)
    parser.add_argument("--nleads", type=int, choices=[1, 2, 3, 4, 6], default=None)
    parser.add_argument("--device", type=int, default=0)
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
        
    # Initialize model
    if args.mode == 'base':
        ckpt = 'checkpoints/camel_base.pt'
    elif args.mode == 'ecgbench':
        ckpt = 'checkpoints/camel_ecginstruct.pt'
    elif args.mode == 'forecast':
        ckpt = 'checkpoints/camel_forecast.pt'
    model = CAMEL(ckpt=ckpt, device=args.device)
    model.run(input_text=args.text, data=args.ecg, args=args)

if __name__ == "__main__":
    main()
