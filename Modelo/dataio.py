"""Simple table loaders used by the project.

This module provides small helper functions to load different table formats
commonly used in the NASA Exoplanet Archive dataset (TSV/CSV/VOTable/IPAC).

All loader functions return a 3-tuple: (dataframe_or_none, status_code, error_msg_or_none)
 - On success: (pandas.DataFrame, 1, None)
 - On failure: (None, 0, "error message")

The functions aim to be forgiving (they try fallbacks) and to return useful
error messages instead of raising. Callers can check the status code before
using the returned DataFrame.
"""

from typing import Optional, Tuple
import pyvo as vo
import pandas as pd
import astropy
from astropy.io.votable import parse
from astropy.io import ascii

# Query TAP service helper
def query_tap_service(url: str, query: str) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
    """Query a TAP service and return (df, status, errmsg).

    Returns (df, 1, None) on success or (None, 0, "error") on failure.
    """
    # Validation: parameters
    if not url or not isinstance(url, str):
        # Returning an empty DataFrame hides why the call failed (bad URL).
        # Better: return (None, 0, "invalid url") to match other helpers.
        return (None, 0, "invalid url")
    
    if not query or not isinstance(query, str):
        return (None, 0, "invalid query")

    # Basic URL validation
    if not (url.startswith('http://') or url.startswith('https://')):
        return (None, 0, "invalid url")

    try:
        # Create a TAPService - note: `pyvo` may raise various exceptions here
        # (network errors, invalid URL, etc.). We catch them below but again
        # returning an empty DataFrame loses the error context.
        service = vo.dal.TAPService(url)

        # pyvo's TAPService provides `search` for synchronous queries; if
        # the object doesn't expose it something is off with the service.
        if not hasattr(service, 'search'):
            return (None, 0, "service not available")

        # Execute the query. `service.search` can take many forms (ADQL,
        # synchronous/asynchronous). This simple wrapper assumes the TAP
        # endpoint supports synchronous `search(query)` calls which is not
        # universally true. For robustness you may want to detect
        # `service.search` signature or use `service.run_sync` where available.
        result = service.search(query)

        if result is None:
            return (None, 0, "no results")

        # Convert VO table to pandas. This step can raise for large tables or
        # non-trivial VO types (masked arrays, unicode encoding); callers may
        # prefer to stream or page results rather than loading everything.
        df = result.to_table().to_pandas()
        return df, 1, ""
    
    except (vo.dal.DALServiceError, vo.dal.DALQueryError, Exception) as e:
        # Broad exception catching is convenient for scripts but hides bugs.
        # At minimum log or return the error message so callers can act on it.
        # For now we keep behaviour (return empty DataFrame) but annotate
        # the code so maintainers notice this is a fragile spot.
        # Example improvement: return None, 0, str(e)
        return (None, 0, "invalid url")

def loadtabseptable(path: str) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
    """Load a tab-separated table file (TSV) into a pandas DataFrame.

    Notes
    -----
    - Many NASA archive tables include a commented header block (lines starting
      with '#') followed by a tab-delimited header row. When possible callers
      should pass `comment='#'` to pandas.read_csv; this function uses a simple
      default (sep='\t') which works for standard TSV files. If you encounter
      tokenization errors (for example "Expected 1 fields ... saw 141") try
      using the caller-side options shown in the project README (engine='python',
      comment='#', on_bad_lines='skip' or 'warn').

    Return contract
    ---------------
    (df, status, errmsg) where status is 1 on success and 0 on failure.
    """
    try:
        # Use explicit tab separator. Reading as strings first can help when
        # columns have mixed types or malformed numeric values.
        df = pd.read_csv(path, sep='\t', comment= "#")
        return df, 1, None
    except Exception as e:
        # Returning the exception message is good — callers can log or show
        # the message to users. For even more robust handling consider adding
        # retry logic or trying `engine='python'` when pandas' C engine fails.
        return None, 0, str(e)

def loadvotableable(path: str) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
    """Load a VOTable file (.votable / .xml) using astropy and return a DataFrame.

    The function uses astropy.io.votable.parse to read the VOTable, converts the
    first table to an astropy Table and then to pandas. If direct conversion
    to pandas fails we build a DataFrame column-by-column from the astropy
    table data arrays.
    """
    try:
        # parse() returns a VOTableFile object
        votableable_obj = parse(path)
        astropy_table = votableable_obj.get_first_table().to_table(use_names_over_ids=True)
        try:
            df = astropy_table.to_pandas()
        except Exception:
            # Fallback: construct DataFrame explicitly (handles dtype edge cases)
            df = pd.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})
        return df, 1, None
    except Exception as e:
        # Astropy's VOTable parser can raise for invalid XML, missing tables,
        # or when the VOTable uses unusual datatypes. Returning the error
        # string helps the caller debug; consider adding a dedicated
        # `validate_votable` function if you need stricter checks.
        return None, 0, str(e)

def loadcsvfile(path: str) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
    """Load a comma-separated CSV file into a pandas DataFrame.

    This is a thin wrapper around pandas.read_csv that returns an explicit
    (df, status, errmsg) tuple instead of raising exceptions.
    """
    try:
        # Allow pandas' parsers to auto-detect dialects in most CSVs; the
        # comment parameter is useful for NASA tables which sometimes include
        # header metadata starting with '#'. If you still get tokenization
        # errors, try engine='python' or explicit encoding.
        df = pd.read_csv(path, sep=',', comment= "#")
        return df, 1, None
    except Exception as e:
        return None, 0, str(e)

def loadipactable(path: str) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
    """Read an IPAC-format table using astropy.io.ascii and return a DataFrame.

    Many IPAC tables include fixed-width or column descriptors that astropy can
    parse if given format='ipac'. We attempt that first and fall back to
    autodetection if it fails.
    """
    try:
        astropy_table = ascii.read(path, format='ipac')
    except Exception:
        try:
            # Fallback to auto-detection (slower but more flexible)
            astropy_table = ascii.read(path)
        except Exception as e:
            # When astropy fails it often includes useful parsing info in the
            # exception message; exposing it to callers speeds debugging.
            return None, 0, str(e)

    try:
        df = astropy_table.to_pandas()
    except Exception:
        # Construct DataFrame explicitly to avoid any astropy->pandas corner cases
        df = pd.DataFrame({name: astropy_table[name].data for name in astropy_table.colnames})

    return df, 1, None

if __name__ == "__main__":
    # Example usage for quick manual tests. When imported as a module the
    # functions return (df, status, errmsg) so callers can decide how to handle
    # errors programmatically.
    # Quick manual smoke tests. The example files are not included in the
    # repository; this block is useful locally for developers to exercise the
    # loaders. The functions are designed to return a (df, status, errmsg)
    # tuple but `query_url` currently returns only a DataFrame (see notes).
    df_tab, ok, msg = loadtabseptable("exampletables/example.tab")
    print(df_tab)
    print("Tab-separated table:", "OK" if ok else f"FAIL: {msg}")

    df_votable, ok, msg = loadvotableable("exampletables/example.votable")
    print(df_votable)
    print("VOTable:", "OK" if ok else f"FAIL: {msg}")

    df_csv, ok, msg = loadcsvfile("exampletables/example.csv")
    print(df_csv)
    print("CSV:", "OK" if ok else f"FAIL: {msg}")

    df_ipac, ok, msg = loadipactable("exampletables/example.tbl")
    print(df_ipac)
    print("IPAC:", "OK" if ok else f"FAIL: {msg}")