import argparse
import yaml
from src.cif_extract import run_extraction

def main():
    parser = argparse.ArgumentParser(description="Extract leakage-free crystal/chemical features from CIF")
    parser.add_argument("--cif-dir", required=True, help="Folder path of *.cif/*.mcif files")
    parser.add_argument("--out-csv", required=True, help="Output feature csv path")
    parser.add_argument("--cfg-path", default="configs/feature_cfg.yaml", help="Config yaml path")
    args = parser.parse_args()

    with open(args.cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_extraction(
        cif_dir=args.cif_dir,
        output_path=args.out_csv,
        cfg=config
    )

if __name__ == "__main__":
    main()