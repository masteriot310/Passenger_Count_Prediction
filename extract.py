from supabase import create_client, Client
import os
from typing import List, Dict

# Read Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_KEY must be set as environment variables"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_passenger_counts() -> List[Dict]:
    """
    Fetch timestamp and count columns from the passenger_count table.

    Returns:
        List[Dict]: [
            {"timestamp": "...", "count": int},
            ...
        ]
    """
    response = (
        supabase
        .table("passenger_count")
        .select("timestamp, count")
        .order("timestamp", desc=False)
        .execute()
    )

    if response.data is None:
        return []

    return response.data
