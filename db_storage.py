import os
import psycopg2
import json

# Only warn about DB failures once per session to avoid log spam
_db_warned = False


def store_raw_options_data(db_params, data, now):
    global _db_warned
    if os.environ.get("DB_STORE_ENABLED", "1") == "0":
        return
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO spx_options_data (data, fetched_at) VALUES (%s, %s)",
            (json.dumps(data), now),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        if not _db_warned:
            print(
                f"Database storage unavailable ({e}). "
                "Plots will continue. Set DB_STORE_ENABLED=0 to silence."
            )
            _db_warned = True