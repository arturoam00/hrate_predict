#!/usr/bin/env python3
import os

import pandas as pd
from dotenv import load_dotenv

from utils import concatenate_all, get_exp_start_end


def main() -> None:
    # Load environment variables
    load_dotenv()
    bin_id = os.environ["BIN_ID"]

    # Get start and end datetime of experiment
    base_path = os.path.join("data", bin_id)
    start, end = get_exp_start_end(base_path=base_path)

    # Create date range with custom frequency with the start and end dates of the experiment
    freq = "s"
    date_range = pd.date_range(start=start, end=end, freq=freq)

    # Create output dir if doesn't exist yet
    output_dir = os.path.join("output", bin_id)
    os.makedirs(output_dir, exist_ok=True)

    # Write to .csv file the result of all data merged and aggregated by freq
    output_path = os.path.join(output_dir, "merged.csv")
    print(f"Merging all data files from {base_path} ...")
    concatenate_all(base_path=base_path, date_range=date_range).to_csv(output_path)
    print(f"Results successfully saved to {output_path}")


if __name__ == "__main__":
    main()
