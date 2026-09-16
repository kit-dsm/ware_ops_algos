import pandas as pd
from pathlib import Path

folder_path = Path(r"U:\Diss\Routing_uncertainty\Results\20260107_184159_results")

def find_parquet_files(folder_path):
    return list(folder_path.glob("*.parquet"))

parquet_files = find_parquet_files(folder_path)

for file in parquet_files:
    df = pd.read_parquet(file)
    file_name = file.stem
    xlsx_file_path = folder_path / (file_name + ".xlsx")
    df.to_excel(xlsx_file_path, index=False)
