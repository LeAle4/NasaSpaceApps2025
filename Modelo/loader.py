import pandas
import astropy
from astropy.io.votable import parse
from astropy.io import ascii

def loadtabseptable(path: str) -> pandas.DataFrame:
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
        # pandas.read_table uses '\t' by default, but be explicit
        return pandas.read_table(path, sep='\t')
    except Exception as e:
        raise IOError(f"Could not read tab-separated file from {path}: {e}")

def loadvotable(path: str) -> pandas.DataFrame:
    """Read a VOTable and return a pandas DataFrame.

    Raises IOError on failure.
    """
    try:
        votable_obj = parse(path)
        astropy_table = votable_obj.get_first_table().to_table(use_names_over_ids=True)
        try:
            return astropy_table.to_pandas()
        except Exception:
            # Fallback if astropy lacks to_pandas
            return pandas.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})
    except Exception as e:
        raise IOError(f"Could not read VOTable from {path}: {e}")

def loadcsvfile(path: str) -> pandas.DataFrame:
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
        return pandas.read_csv(path)
    except Exception as e:
        raise IOError(f"Could not read CSV file from {path}: {e}")

def loadipactable(path: str) -> pandas.DataFrame:
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
            raise IOError(f"Could not read IPAC table from {path}: {e}")

    try:
        return astropy_table.to_pandas()
    except Exception:
        return pandas.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})

if __name__ == "__main__":
    # Example usage (for testing purposes)
    try:
        df_tab = loadtabseptable("example_tab.txt")
        print("Tab-separated table loaded successfully.")
    except IOError as e:
        print(e)

    try:
        df_vot = loadvotable("example.vot")
        print("VOTable loaded successfully.")
    except IOError as e:
        print(e)

    try:
        df_csv = loadcsvfile("example.csv")
        print("CSV file loaded successfully.")
    except IOError as e:
        print(e)

    try:
        df_ipac = loadipactable("example.ipac")
        print("IPAC table loaded successfully.")
    except IOError as e:
        print(e)