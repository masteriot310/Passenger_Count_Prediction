from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()

    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}"
        )
