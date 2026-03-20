import pandas as pd
from pathlib import Path

#script pour copier les valeurs des seconds membres des contraintes couplantes de dsr_industrie_stock_lt.txt dans des fichiers tsv energy_max_daily_[area]_dsr_industrie.tsv, en divisant les valeurs par 24 et en dupliquant chaque ligne 24 fois pour correspondre à une valeur par heure.
def process_dsr_files(input_dir: str, output_dir: str):
    """
    Retrieve values from files named '[area]_dsr_industrie_stock_lt.txt' in the input directory,
    and write them to TSV files named 'energy_max_daily_[area]_dsr_industrie.tsv' in the output directory.

    Args:
        input_dir (str): Path to the directory containing the input files.
        output_dir (str): Path to the directory where the output TSV files will be written.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all files matching the pattern
    for file_path in input_path.glob("*_dsr_industrie_stock_lt.txt"):
        # Extract the area from the filename (first part before '_')
        area = file_path.stem.split('_')[0]

        # Read the file assuming it's space-separated values
        # Adjust sep if the file format is different (e.g., ',' for CSV)
        df = pd.read_csv(file_path, sep='\s+', header=None, engine='python')

        # Divide all values by 24
        df = df / 24

        # Duplicate each row 24 times
        df = df.loc[df.index.repeat(24)].reset_index(drop=True)

        # Write to TSV
        output_file = output_path / f"energy_max_daily_{area}_dsr_industrie.tsv"
        df.to_csv(output_file, sep='\t', index=False, header=False)

        print(f"Processed {file_path} -> {output_file}")


# Example usage
if __name__ == "__main__":
    input_directory = r"c:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25\input\bindingconstraints"
    output_directory = r"c:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\input\data-series"
    process_dsr_files(input_directory, output_directory)