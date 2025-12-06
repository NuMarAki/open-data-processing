import os
from glob import glob
from typing import List, Optional
import time

import pandas as pd
from src.data.feature_processing import _clean_col

def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(file_path, **kwargs)

def load_parquet(file_path: str, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(file_path, **kwargs)

def _read_file(path: str, **kwargs) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv',):
        return load_csv(path, **kwargs)
    elif ext in ('.parquet', '.pq'):
        return load_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_data_from_dir(dir_path: str, patterns: Optional[List[str]] = None, file_limit: Optional[int] = None, **read_kwargs) -> pd.DataFrame:
    if patterns is None:
        patterns = ['*']

    if os.path.isfile(dir_path):
        return _read_file(dir_path, **read_kwargs)

    all_frames = []
    for pattern in patterns:
        glob_path = os.path.join(dir_path, pattern)
        matched = sorted(glob(glob_path))
        if file_limit is not None and file_limit > 0:
            allowed = file_limit - len(all_frames)
            if allowed <= 0:
                break
            matched = matched[:allowed]

        total_files = len(matched)
        for i, fp in enumerate(matched, start=1):
            start = time.perf_counter()
            try:
                df = _read_file(fp, **read_kwargs)
                basename = os.path.basename(fp)
                import re
                match = re.search(r'rais_\w+(\d{4})', basename)
                if match:
                    year = int(match.group(1))
                    df['ano'] = year
                else:
                    df['ano'] = None
                all_frames.append(df)
            except Exception as e:
                print(f"Warning: failed to read {fp}: {e}")
                continue
            elapsed = time.perf_counter() - start
            if total_files > 1:
                remaining = total_files - i
                eta = remaining * elapsed
                print(f"Read file {i}/{total_files} ({os.path.basename(fp)}), time={elapsed:.2f}s, ETA~{eta:.1f}s")
            else:
                print(f"Read file {i}/{total_files} ({os.path.basename(fp)}), time={elapsed:.2f}s")

    if not all_frames:
        print(f"No files matched patterns in {dir_path}")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.columns = [_clean_col(c) for c in df.columns]

    print(f"Loaded {len(all_frames)} file(s) from {dir_path} — total rows: {len(df)}")
    return df

def load_data(path: str, patterns: Optional[List[str]] = None, file_limit: Optional[int] = None, **read_kwargs) -> pd.DataFrame:
    return load_data_from_dir(path, patterns=patterns, file_limit=file_limit, **read_kwargs)
