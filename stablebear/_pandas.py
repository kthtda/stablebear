def import_pandas():
    """The pandas module, or an ImportError that says pandas is needed."""
    try:
        import pandas
    except ImportError as error:
        raise ImportError("Converting to pandas needs the pandas package, which is not installed") from error
    return pandas
