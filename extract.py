from supabase import create_client, Client
import os
import pandas as pd
from typing import List, Dict
from env_loader import load_env

# Load environment variables
load_env()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_passenger_counts() -> List[Dict]:
    """
    Fetch timestamp and count columns from the passenger_count table.
    """
    response = (
        supabase
        .table("passenger_count")
        .select("timestamp, count")
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data or []
import pandas as pd

def fetch_passenger_counts_df() -> pd.DataFrame:
    """
    Fetch passenger counts as a Pandas DataFrame
    """
    data = fetch_passenger_counts()
    df = pd.DataFrame(data)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.sort_values("timestamp").reset_index(drop=True)

if __name__ == "__main__":
    data = fetch_passenger_counts()

    print(f"Rows fetched: {len(data)}")

    # Print first 10 rows only (avoid dumping huge tables)
    for row in data[:10]:
        print(row)
