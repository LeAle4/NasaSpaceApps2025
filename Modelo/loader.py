import pandas
import astropy
from astropy.io.votable import parse
from astropy.io import ascii

def loadtabseptable(path: str):
    """Load a tab-separated table file and return a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the tab-separated table file.

    Returns
    -------
    pandas.DataFrame
        Parsed table contents.

    Raises
    ------
    IOError
        If pandas fails to read the file.
    """
    try:
        df = pandas.read_table(path, sep='\t')
        return df, 1, None
    except Exception as e:
        return None, 0, str(e)

def loadvotableable(path: str):
    """Read a votableable and return a pandas DataFrame.

    Raises IOError on failure.
    """
    try:
        votableable_obj = parse(path)
        astropy_table = votableable_obj.get_first_table().to_table(use_names_over_ids=True)
        try:
            df = astropy_table.to_pandas()
        except Exception:
            df = pandas.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})
        return df, 1, None
    except Exception as e:
        return None, 0, str(e)

def loadcsvfile(path: str):
    """Load a CSV file into a pandas DataFrame with a clear error on failure.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Parsed CSV contents.

    Raises
    ------
    IOError
        If pandas fails to read the CSV file.
    """
    try:
        df = pandas.read_csv(path)
        return df, 1, None
    except Exception as e:
        return None, 0, str(e)

def loadipactable(path: str):
    """Read an IPAC-format table and return a pandas DataFrame.

    Tries astropy.io.ascii with format='ipac' then falls back to auto-detect.
    Raises IOError on failure.
    """
    try:
        astropy_table = ascii.read(path, format='ipac')
    except Exception:
        try:
            astropy_table = ascii.read(path)
        except Exception as e:
            return None, 0, str(e)

    try:
        df = astropy_table.to_pandas()
    except Exception:
        df = pandas.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})

    return df, 1, None

if __name__ == "__main__":
    # Example usage (for testing purposes)
    df_tab, ok, msg = loadtabseptable("example.tab")
    print("Tab-separated table:", "OK" if ok else f"FAIL: {msg}")

    df_votable, ok, msg = loadvotableable("example.votable")
    print("votableable:", "OK" if ok else f"FAIL: {msg}")

    df_csv, ok, msg = loadcsvfile("example.csv")
    print(df_csv)
    print("CSV:", "OK" if ok else f"FAIL: {msg}")

    df_ipac, ok, msg = loadipactable("example.ipac")
    print("IPAC:", "OK" if ok else f"FAIL: {msg}")