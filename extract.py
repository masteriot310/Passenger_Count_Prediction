from supabase import create_client, Client
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime
from env_loader import load_env

# =========================
# ENV SETUP
# =========================
load_env()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# CACHE SETTINGS
# =========================
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "passenger_count.parquet")

os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# DATABASE FETCH
# =========================
def fetch_passenger_counts(since: str | None = None) -> List[Dict]:
    """
    Fetch timestamp and count columns from the passenger_count table.
    If `since` is provided, only fetch rows newer than that timestamp.
    """
    query = (
        supabase
        .table("passenger_count")
        .select("timestamp, count")
        .order("timestamp", desc=False)
    )

    if since:
        query = query.gt("timestamp", since)

    response = query.execute()
    return response.data or []

# =========================
# CACHE-AWARE FETCH
# =========================
def fetch_passenger_counts_df(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch passenger counts as a Pandas DataFrame using local cache.
    Only new rows are pulled from Supabase to reduce latency.
    """

    # -------------------------
    # Load cache if available
    # -------------------------
    if os.path.exists(CACHE_FILE) and not force_refresh:
        df_cache = pd.read_parquet(CACHE_FILE)
        df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"])

        last_ts = df_cache["timestamp"].max().isoformat()
    else:
        df_cache = pd.DataFrame(columns=["timestamp", "count"])
        last_ts = None

    # -------------------------
    # Fetch only new rows
    # -------------------------
    new_data = fetch_passenger_counts(since=last_ts)

    if new_data:
        df_new = pd.DataFrame(new_data)
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

        df = pd.concat([df_cache, df_new], ignore_index=True)
        df = df.drop_duplicates(subset="timestamp")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Save updated cache
        df.to_parquet(CACHE_FILE)

        return df

    # No new data → return cache
    return df_cache.sort_values("timestamp").reset_index(drop=True)

# =========================
# LATENCY-ROBUST TIMING
# =========================
def compute_latency_metrics(df: pd.DataFrame) -> dict:
    """
    Compute latency-resilient timing metrics (4G safe).
    """
    if len(df) < 2:
        return {}

    deltas = df["timestamp"].diff().dt.total_seconds().dropna()

    q1 = deltas.quantile(0.25)
    q3 = deltas.quantile(0.75)
    iqr = q3 - q1

    filtered = deltas[
        (deltas >= q1 - 1.5 * iqr) &
        (deltas <= q3 + 1.5 * iqr)
    ]

    return {
        "raw_avg_seconds": round(deltas.mean(), 4),
        "median_seconds": round(deltas.median(), 4),
        "clean_avg_seconds": round(filtered.mean(), 4),
        "samples_used": len(filtered),
        "total_samples": len(deltas)
    }

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = fetch_passenger_counts_df()

    print(f"Total rows (cached): {len(df)}")

    metrics = compute_latency_metrics(df)

    if metrics:
        print("\nLatency Metrics (Network-Adjusted):")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    print("\nFirst 10 rows:")
    print(df.head(10))
