# Code snippets

Small helpers I might promote into the library later.

---

## IO helpers

```python
from pathlib import Path
from typing import Union

import polars as pl
import pandas as pd

DataFrameLike = Union[pl.DataFrame, pd.DataFrame]


def save_result(df: DataFrameLike, path: Path, fmt: str = "parquet") -> Path:
    """
    Save a Polars or pandas DataFrame to `path` (parquet or csv).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        if isinstance(df, pl.DataFrame):
            df.write_parquet(path)
        else:
            df.to_parquet(path, index=False)
    elif fmt == "csv":
        if isinstance(df, pl.DataFrame):
            df.write_csv(path)
        else:
            df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")

    return path