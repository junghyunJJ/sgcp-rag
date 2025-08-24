#!/usr/bin/env python3
"""Helper script to get a Supabase access token for MCP server configuration.

This script allows you to sign in and get a JWT token that can be used
with the MCP servers.
"""

import os
import sys
from getpass import getpass

import requests
from dotenv import load_dotenv

load_dotenv()

# Get Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    sys.exit(1)


def sign_in(email: str, password: str):
    """Sign in to Supabase and get access token."""
    try:
        # Call the LangConnect API signin endpoint
        response = requests.post(
            f"{API_BASE_URL}/auth/signin",
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("access_token"), data.get("refresh_token")
        error = response.json()
        print(f"Sign in failed: {error.get('detail', 'Unknown error')}")
        return None, None

    except Exception as e:
        print(f"Error during sign in: {e!s}")
        return None, None


def test_token(token: str):
    """Test if the token works by calling the collections endpoint."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/collections", headers={"Authorization": f"Bearer {token}"}
        )
        return response.status_code == 200
    except:
        return False


def main():
    print("LangConnect MCP Access Token Helper")
    print("=" * 40)
    print()

    # Get credentials
    email = input("Enter your email: ")
    password = getpass("Enter your password: ")

    print("\nSigning in...")
    access_token, refresh_token = sign_in(email, password)

    if access_token:
        print("\n✅ Sign in successful!")
        print("\nTesting token...")

        if test_token(access_token):
            print("✅ Token is valid and working!")

            print("\n" + "=" * 60)
            print("YOUR ACCESS TOKEN:")
            print("=" * 60)
            print(access_token)
            print("=" * 60)

            print("\nTo use this token:")
            print("\n1. For MCP stdio server, update mcp_config.json:")
            print(f'   "SUPABASE_JWT_SECRET": "{access_token}"')

            print("\n2. For MCP SSE server, set environment variable:")
            print(f'   export SUPABASE_JWT_SECRET="{access_token}"')

            print("\n3. Or add to .env file:")
            print(f"   SUPABASE_JWT_SECRET={access_token}")

            print("\n⚠️  Note: This token will expire in about 1 hour.")
            print("    Run this script again to get a new token when needed.")

            if refresh_token:
                print(f"\nRefresh token (for future use): {refresh_token}")
        else:
            print("❌ Token validation failed. The token may be invalid.")
    else:
        print("\n❌ Sign in failed. Please check your credentials.")


if __name__ == "__main__":
    main()
