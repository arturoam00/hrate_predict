import os

import pandas as pd
from dotenv import load_dotenv

from utils import AccelerometerLoader, get_exp_start_end


def main():
    # load environment variables
    load_dotenv()

    # Get start and end datetime of experiment
    base_path = os.path.join("data", os.environ["BIN_ID"])
    start, end = get_exp_start_end(base_path=base_path)

    # Load Accelerometer.csv data and aggregate data by 1 sec. frequency
    freq = "s"
    date_range = pd.date_range(start=start, end=end, freq=freq)
    accelerometer_loader = AccelerometerLoader(
        base_path=base_path, date_range=date_range
    )

    acclerometer_df = accelerometer_loader.load()
    print("Data loaded and transformed\n")
    return acclerometer_df


if __name__ == "__main__":
    main()
