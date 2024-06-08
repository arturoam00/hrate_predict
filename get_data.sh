#!/bin/bash

# Get the bin id 
source .env

# Base URL
base_url="https://filebin.net/$BIN_ID/"

files=(
    "Accelerometer.csv"
    "Barometer.csv"
    "Gyroscope.csv"
    "Linear_Accelerometer.csv"
    "Location.csv"
    "Magnetometer.csv"
    "Proximity.csv"
    "time.csv" 

)

# Create the data directory if it doesn't exist
mkdir -p data/$BIN_ID

# Loop over the names and use wget to download each file to the 1data directory
for file in "${files[@]}"
do
    url="${base_url}${file}"
    echo "Downloading $url to data/$BIN_ID ..."
    wget -NP data/$BIN_ID "$url"
done

echo "All downloads completed."

