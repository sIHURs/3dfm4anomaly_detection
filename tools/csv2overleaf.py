import pandas as pd

dir_path = "results"
df = pd.read_csv(f"{dir_path}/Rotation_error.csv")

# save as latex
latex = df.to_latex(index=False, escape=True, float_format="%.3f")

latex_path = f"{dir_path}/Rotation_error.tex"
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex)