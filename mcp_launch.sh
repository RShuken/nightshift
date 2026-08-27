#!/bin/bash
# Launcher for the nightshift MCP server — pins cwd so imports and data paths resolve.
cd /Users/shuken/AI/dispered
exec python3 -m nightshift.mcp_server
