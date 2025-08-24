#!/bin/bash
# Script to run tests without OpenAI API key in environment

# Backup .env if it exists
if [ -f .env ]; then
    cp .env .env.backup
    # Remove or comment out OPENAI_API_KEY
    sed -i '' 's/^OPENAI_API_KEY=/#OPENAI_API_KEY=/' .env
fi

# Run tests
IS_TESTING=true uv run pytest "$@"
TEST_EXIT_CODE=$?

# Restore .env if we backed it up
if [ -f .env.backup ]; then
    mv .env.backup .env
fi

exit $TEST_EXIT_CODE