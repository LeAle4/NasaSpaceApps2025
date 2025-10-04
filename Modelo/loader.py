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
        return df, 1
    except Exception:
        return None, 0

def loadvotable(path: str):
    """Read a VOTable and return a pandas DataFrame.

    Raises IOError on failure.
    """
    try:
        votable_obj = parse(path)
        astropy_table = votable_obj.get_first_table().to_table(use_names_over_ids=True)
        try:
            df = astropy_table.to_pandas()
        except Exception:
            df = pandas.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})
        return df, 1
    except Exception:
        return None, 0

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
        return df, 1
    except Exception:
        return None, 0

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
        except Exception:
            return None, 0

    try:
        df = astropy_table.to_pandas()
    except Exception:
        df = pandas.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})

    return df, 1

if __name__ == "__main__":
    # Example usage (for testing purposes)
    df_tab, ok = loadtabseptable("example_tab.txt")
    print("Tab-separated table:", "OK" if ok else "FAIL")

    df_vot, ok = loadvotable("example.vot")
    print("VOTable:", "OK" if ok else "FAIL")

    df_csv, ok = loadcsvfile("example.csv")
    print("CSV:", "OK" if ok else "FAIL")

    df_ipac, ok = loadipactable("example.ipac")
    print("IPAC:", "OK" if ok else "FAIL")