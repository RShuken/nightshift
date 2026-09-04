#!/bin/bash
# Launcher for the nightshift MCP server — pins cwd to the repo so imports and data paths resolve.
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0")")"
exec python3 -m nightshift.mcp_server
