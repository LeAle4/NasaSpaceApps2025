import pandas
import astropy
from astropy.io.votable import parse
from astropy.io import ascii

def loadtabseptable(path: str):
    """Load a tab-separated file and return (df, status).

    Returns (DataFrame, 1) on success or (None, 0) on failure.
    """
    try:
        df = pandas.read_table(path, sep='\t')
        return df, 1
    except Exception:
        return None, 0

def loadvotable(path: str):
    """Read a VOTable and return (df, status).

    Returns (DataFrame, 1) on success or (None, 0) on failure.
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
    """Load a CSV file and return (df, status).

    Returns (DataFrame, 1) on success or (None, 0) on failure.
    """
    try:
        df = pandas.read_csv(path)
        return df, 1
    except Exception:
        return None, 0

def loadipactable(path: str):
    """Read an IPAC-format table and return (df, status).

    Tries astropy.io.ascii with format='ipac' then falls back to auto-detect.
    Returns (DataFrame, 1) on success or (None, 0) on failure.
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

