import os
import re
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse image ROCAUC from experiment logs and build a table."
    )
    parser.add_argument(
        "--log_root",
        type=str,
        required=True,
        help="Root log directory, e.g. logs/"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="rocauc_table.csv",
        help="Output CSV file name"
    )
    parser.add_argument(
        "--out_xlsx",
        type=str,
        default=None,
        help="Optional output Excel file name"
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="image ROCAUC",
        help="Metric key to search in logs (default: 'image ROCAUC')"
    )
    parser.add_argument(
        "--take_last",
        action="store_true",
        help="If set, take the last occurrence of the metric in each log (default: first)"
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Decimal places for rounding (default: 1)"
    )
    return parser.parse_args()


def extract_metric(log_path, pattern, take_last=False):
    """Extract metric value from a log file and return percentage."""
    val = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                val = float(m.group(1))
                if not take_last:
                    break
    if val is None:
        return None
    return val * 100.0


def main():
    args = parse_args()

    # regex: e.g. image ROCAUC: 0.780
    pattern = re.compile(
        rf"{re.escape(args.metric_name)}\s*:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE
    )

    records = []

    # iterate experiments
    for exp_name in sorted(os.listdir(args.log_root)):
        exp_dir = os.path.join(args.log_root, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        # iterate class logs
        for fname in sorted(os.listdir(exp_dir)):
            if not fname.endswith(".log"):
                continue

            class_name = os.path.splitext(fname)[0]
            log_path = os.path.join(exp_dir, fname)

            val = extract_metric(
                log_path,
                pattern,
                take_last=args.take_last
            )
            if val is None:
                print(f"[WARN] Metric not found in {log_path}")
                continue

            records.append({
                "class": class_name,
                "experiment": exp_name,
                "value": val,
            })

    if not records:
        raise RuntimeError("No metrics found. Check log_root or metric_name.")

    df_long = pd.DataFrame(records)

    # pivot to wide table
    table = df_long.pivot(
        index="class",
        columns="experiment",
        values="value"
    ).sort_index()

    table = table.round(args.round)

    # save outputs
    table.to_csv(args.out_csv)
    print(f"Saved CSV: {args.out_csv}")

    if args.out_xlsx is not None:
        table.to_excel(args.out_xlsx)
        print(f"Saved XLSX: {args.out_xlsx}")

    print("\nResult table:")
    print(table)


if __name__ == "__main__":
    main()
