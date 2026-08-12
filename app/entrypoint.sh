#!/bin/sh
set -e

if [ "$1" = "serve" ]; then
    shift
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8080 "$@"