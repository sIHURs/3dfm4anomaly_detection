import os
import re
import argparse
import pandas as pd


DEFAULT_METRICS = [
    "image ROCAUC",
    "pixel ROCAUC",
    "aupro",
    "au_roc",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse multiple metrics from experiment logs and build tables."
    )
    parser.add_argument(
        "--log_root",
        type=str,
        required=True,
        help="Root log directory, e.g. logs/"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="metrics_table",
        help="Output prefix for CSV files (default: metrics_table)"
    )
    parser.add_argument(
        "--out_xlsx",
        type=str,
        default=None,
        help="Optional output Excel file name (all metrics in one workbook)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric keys to search (default: image ROCAUC,pixel ROCAUC,aupro,au_roc)"
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
    parser.add_argument(
        "--no_percent",
        action="store_true",
        help="If set, do NOT multiply values by 100 (default: multiply by 100)"
    )
    return parser.parse_args()


def build_pattern(metric_name: str) -> re.Pattern:
    """
    Match lines like:
      image ROCAUC: 0.935
      pixel ROCAUC: 0.996
      aupro: 0.9487
      other au_roc: 0.9268
    We allow optional words before the metric (e.g. 'other ') and tolerate spaces/underscores.
    """
    escaped = re.escape(metric_name)
    return re.compile(
        rf"(?:^|\b).*{escaped}\s*:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE
    )


def extract_metric(log_path: str, pattern: re.Pattern, take_last: bool = False):
    """Extract metric value from a log file."""
    val = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                if not take_last:
                    break
    return val


def safe_sheet_name(name: str) -> str:
    """Excel sheet name constraints: max 31 chars, no []:*?/\\ """
    bad = r'[]:*?/\\'
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip()
    if len(name) > 31:
        name = name[:31]
    if not name:
        name = "sheet"
    return name


def make_table(records, round_n: int):
    """
    records: list of dicts {class, experiment, value}
    return pivot table with mean row appended.
    """
    df_long = pd.DataFrame(records)
    table = df_long.pivot(index="class", columns="experiment", values="value").sort_index()
    table = table.round(round_n)

    mean_row = table.mean(axis=0).round(round_n)
    mean_row.name = "mean"
    table = pd.concat([table, mean_row.to_frame().T])
    return table


def main():
    args = parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics provided. Use --metrics.")

    patterns = {m: build_pattern(m) for m in metrics}

    # metric -> records list
    all_records = {m: [] for m in metrics}

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

            # extract each metric from this log
            for metric_name, pattern in patterns.items():
                val = extract_metric(log_path, pattern, take_last=args.take_last)
                if val is None:
                    # don't spam too hard; you can uncomment the next line if you want warnings
                    # print(f"[WARN] '{metric_name}' not found in {log_path}")
                    continue

                if not args.no_percent:
                    val = val * 100.0

                all_records[metric_name].append({
                    "class": class_name,
                    "experiment": exp_name,
                    "value": val,
                })

    # build tables
    tables = {}
    for metric_name, records in all_records.items():
        if not records:
            print(f"[WARN] No records found for metric '{metric_name}'. Skipping.")
            continue
        tables[metric_name] = make_table(records, args.round)

        out_csv = f"{args.out_prefix}_{metric_name.replace(' ', '_')}.csv"
        tables[metric_name].to_csv(out_csv)
        print(f"Saved CSV: {out_csv}")

    if not tables:
        raise RuntimeError("No metrics found at all. Check log_root / metric keys.")

    # optional: write one xlsx with multiple sheets
    if args.out_xlsx is not None:
        with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
            for metric_name, table in tables.items():
                sheet = safe_sheet_name(metric_name)
                table.to_excel(writer, sheet_name=sheet)
        print(f"Saved XLSX: {args.out_xlsx}")

    # print a quick preview
    print("\nPreview (first metric table):")
    first_metric = next(iter(tables.keys()))
    print(f"== {first_metric} ==")
    print(tables[first_metric])


if __name__ == "__main__":
    main()
