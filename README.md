For the `get_data.sh` script to work, save an environment variable `BIN_ID` in a `.env` file, like

```
# .env
BIN_ID=blabei80o1zfv43
```

Then just run it
```
./get_data.sh
```

After this, one can run `main.py` just like (*with the corresponding virtual environment activated*)

```
./main.py
```

So far this just merges all the downloaded data files in one big `.csv` with sensible names for the feature columns and with the data aggregated by frequency of 1 second. Check `main.py` to change this last thing.

The results should be available in an `output/` folder.