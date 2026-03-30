#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


# CLASS_ORDER = [
#     "01Gorilla",
#     "02Unicorn",
#     "03Mallard",
#     "04Turtle",
#     "05Whale",
#     "06Bird",
#     "07Owl",
#     "08Sabertooth",
#     "09Swan",
#     "10Sheep",
#     "11Pig",
#     "12Zalika",
#     "13Pheonix",
#     "14Elephant",
#     "15Parrot",
#     "16Cat",
#     "17Scorpion",
#     "18Obesobeso",
#     "19Bear",
#     "20Puppy",
# ]

CLASS_ORDER = [
    "binderclip",
    "binderclip2",
    "bowl_upright",
    "box",
    "can",
    "charger",
    "cup1_upright",
    "cup2_upright",
    "cup2_upright2",
    "cup2_upright3",
    "gluebottle",
    "gluebottle2",
    "phonecase",
    "phonecase2",
    "rubberduck",
    "spoon_upright",
    "spraybottle2",
    "tennisball",
]


DEFAULT_CAPTION = "Results over all classes in the MAD-Sim dataset"
DEFAULT_LABEL = "tab:results_mad"


def format_value(x, digits=3):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def escape_latex(text: str) -> str:
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def find_class_col(df: pd.DataFrame, user_class_col: str | None = None) -> str:
    if user_class_col is not None:
        if user_class_col not in df.columns:
            raise ValueError(
                f"Class column '{user_class_col}' not found.\nAvailable columns: {list(df.columns)}"
            )
        return user_class_col

    for c in ["Class", "class"]:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find class column. Expected one of ['Class', 'class'].\n"
        f"Available columns: {list(df.columns)}"
    )


def parse_cols_arg(cols_arg: str | None, df: pd.DataFrame, class_col: str) -> list[str]:
    """
    Fully custom columns:
    --cols ate_mean,ate_median,ate_max
    """
    if cols_arg is None or not cols_arg.strip():
        # 默认使用除 class_col 外所有列
        return [c for c in df.columns if c != class_col]

    cols = [c.strip() for c in cols_arg.split(",") if c.strip()]
    if not cols:
        raise ValueError("Empty --cols provided.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"These columns were not found in CSV: {missing}\nAvailable columns: {list(df.columns)}"
        )

    if class_col in cols:
        raise ValueError(
            f"Class column '{class_col}' should not be included in --cols."
        )

    return cols


def parse_header_names_arg(header_names_arg: str | None, cols: list[str]) -> list[str]:
    """
    Optional display names for headers:
    --header-names Mean,Median,Max,p90,p95
    """
    if header_names_arg is None or not header_names_arg.strip():
        return cols

    names = [x.strip() for x in header_names_arg.split(",")]
    if len(names) != len(cols):
        raise ValueError(
            f"--header-names length ({len(names)}) must match number of --cols ({len(cols)})."
        )
    return names


def compute_overall_row(df: pd.DataFrame, cols: list[str], class_col: str, digits: int) -> dict:
    overall_names = {"mean", "overall", "avg", "average", "all"}
    overall_row = None

    for _, row in df.iterrows():
        name = str(row[class_col]).strip().lower()
        if name in overall_names:
            overall_row = row
            break

    if overall_row is not None:
        return {col: format_value(overall_row[col], digits) for col in cols}

    df_classes = df[df[class_col].astype(str).str.strip().isin(CLASS_ORDER)]
    result = {}
    for col in cols:
        numeric_series = pd.to_numeric(df_classes[col], errors="coerce")
        result[col] = format_value(numeric_series.mean(), digits)
    return result


def build_alignment(n_cols: int) -> str:
    return "l " + " ".join(["r"] * n_cols)


def build_table(
    df: pd.DataFrame,
    cols: list[str],
    header_names: list[str],
    class_col: str | None = None,
    digits: int = 3,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
    overall_name: str = "mean",
    add_section: bool = False,
    section_title: str = "Results on dataset RAD",
    section_label: str = "sec:results_rad",
):
    class_col = find_class_col(df, class_col)

    df = df.copy()
    df[class_col] = df[class_col].astype(str).str.strip()

    row_map = {row[class_col]: row for _, row in df.iterrows()}
    overall = compute_overall_row(df, cols, class_col, digits)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(rf"\begin{{tabular}}{{{build_alignment(len(cols))}}}")
    lines.append(r"\toprule")

    header = [r"\textbf{Class}"] + [rf"\textbf{{{escape_latex(h)}}}" for h in header_names]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for cls in CLASS_ORDER:
        row = row_map.get(cls, None)
        values = [escape_latex(cls)]
        for col in cols:
            if row is None:
                values.append("")
            else:
                values.append(format_value(row[col], digits))
        lines.append(" & ".join(values) + r" \\")

    lines.append(r"\midrule")
    overall_values = [escape_latex(overall_name)] + [overall[col] for col in cols]
    lines.append(" & ".join(overall_values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{escape_latex(caption)}}}")
    lines.append(rf"\label{{{escape_latex(label)}}}")
    lines.append(r"\end{table}")

    if add_section:
        lines.append("")
        lines.append(rf"\section{{{escape_latex(section_title)}}}")
        lines.append(rf"\label{{{escape_latex(section_label)}}}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Overleaf LaTeX table from CSV with fully custom column names."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV file")
    parser.add_argument(
        "--class-col",
        type=str,
        default=None,
        help="Class column name. Default: auto-detect Class/class.",
    )
    parser.add_argument(
        "--cols",
        type=str,
        default=None,
        help=(
            "Comma-separated CSV columns to include, in desired order. "
            "Example: --cols ate_mean,ate_median,ate_max,ate_p90,ate_p95"
        ),
    )
    parser.add_argument(
        "--header-names",
        type=str,
        default=None,
        help=(
            "Optional comma-separated display names for table header. "
            "Example: --header-names Mean,Median,Max,p90,p95"
        ),
    )
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--caption", type=str, default=DEFAULT_CAPTION)
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL)
    parser.add_argument(
        "--overall-name",
        type=str,
        default="mean",
        help="Name of the final summary row, e.g. mean / overall / avg",
    )
    parser.add_argument(
        "--add-section",
        action="store_true",
        help="Append a section after the table.",
    )
    parser.add_argument(
        "--section-title",
        type=str,
        default="Results on dataset RAD",
    )
    parser.add_argument(
        "--section-label",
        type=str,
        default="sec:results_rad",
    )
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    class_col = find_class_col(df, args.class_col)
    cols = parse_cols_arg(args.cols, df, class_col)
    header_names = parse_header_names_arg(args.header_names, cols)

    latex = build_table(
        df=df,
        cols=cols,
        header_names=header_names,
        class_col=class_col,
        digits=args.digits,
        caption=args.caption,
        label=args.label,
        overall_name=args.overall_name,
        add_section=args.add_section,
        section_title=args.section_title,
        section_label=args.section_label,
    )

    if args.output:
        args.output.write_text(latex, encoding="utf-8")
        print(f"[OK] Written to {args.output}")
    else:
        print(latex)


if __name__ == "__main__":
    main()