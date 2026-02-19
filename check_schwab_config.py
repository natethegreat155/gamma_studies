#!/usr/bin/env python3
"""Verify Schwab API configuration before running main.py."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.environ.get("SCHWAB_API_KEY")
app_secret = os.environ.get("SCHWAB_APP_SECRET")
redirect_uri = os.environ.get("SCHWAB_REDIRECT_URI", "https://127.0.0.1")

def has_hidden_chars(s: str) -> bool:
    return any(ord(c) < 32 or ord(c) > 126 for c in s if c not in "\n\r\t")

print("Schwab configuration check:")
print("-" * 50)
print("  Mapping: SCHWAB_API_KEY = App Key (from portal)")
print("           SCHWAB_APP_SECRET = Secret (from portal)")
print()
print(f"  SCHWAB_API_KEY:    {'LOADED' if api_key and api_key != 'YOUR_SCHWAB_CLIENT_ID@AMER.OAUTHAP' else 'MISSING or placeholder'}")
if api_key:
    print(f"    Length: {len(api_key)} chars  Ends with @AMER.OAUTHAP: {'@AMER.OAUTHAP' in api_key}")
    if has_hidden_chars(api_key):
        print("    WARNING: Contains hidden/special characters - try copying again")
print(f"  SCHWAB_APP_SECRET: {'LOADED' if app_secret and app_secret != 'YOUR_APP_SECRET_HERE' else 'MISSING or placeholder'}")
if app_secret:
    print(f"    Length: {len(app_secret)} chars")
    if has_hidden_chars(app_secret):
        print("    WARNING: Contains hidden/special characters - try copying again")
print(f"  Callback URL:      {redirect_uri}")
print()
if api_key and "YOUR_" in api_key:
    print("ERROR: App Key not set. Put your App Key from the portal into SCHWAB_API_KEY in .env")
elif app_secret and "YOUR_" in app_secret:
    print("ERROR: Secret not set. Put your Secret from the portal into SCHWAB_APP_SECRET in .env")
else:
    print("Credentials OK. If you still get 'invalid_client':")
    print("  1. Portal Environment: Is your app in 'Production'? (schwab-py uses production API)")
    print("  2. App Status: Must be 'Ready for Use' (not Pending/Development)")
    print("  3. Regenerate Secret: In the portal, regenerate the Secret and paste the new one")
    print("  4. Re-copy App Key: Copy again (no leading/trailing spaces)")
    print("  5. Contact Schwab: traderapi@schwab.com if nothing works")
