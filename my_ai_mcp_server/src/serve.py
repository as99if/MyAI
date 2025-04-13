
from mcp.server.fastmcp import FastMCP
from src.tools.gemini import gemini_inference

mcp = FastMCP("MyAIMCPServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# Expose the tools
# mcp.expose(gemini_inference)

mcp.add_tool(gemini_inference)

if __name__ == "__main__":
    mcp.run(transport="sse")