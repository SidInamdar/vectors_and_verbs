---
title: "USB-C of AI Space: The Model Context Protocol"
date: 2026-01-15T20:58:53+05:30
draft: true
tags: ["mcp", "anthropic", "agentic-workflows", "LLM-orchestration", "artificial-intelligence", "standardization", "JSON-RPC", "SSE", "fastmcp"]
categories: ["artificial-intelligence", "large-language-models"]
math: true
summary: "MCP is the open standard for connecting AI models to data and tools. Discover how Anthropic's new protocol solves the $N \times M$ integration problem, creating a plug-and-play ecosystem for AI agents."
---

From 2022 to 2024, the world was abuzz with the technical explosion of AI. Something that was once a niche mathematical/statistical domain was suddenly being sold as a consumer app. The hype was palpable, and the potential was undeniable.

However, as the AI landscape evolved, it became clear that the promise of AI was not just about the models themselves, but about how they interacted with the world around them. This led to multiple competitors developing variants of their own solutions for AI-embedded ecosystems, trying to capitalize on market share based solely on their own tools and proprietary protocols.

This led to a fragmented solution space where learning more jargon was a prerequisite for usage. This is often referred to as the **$N \times M$ problem**, where $N$ is the number of models and $M$ is the number of tools.

In 2024, an open standard movement led by **Anthropic** introduced the **Model Context Protocol (MCP)**. This created a uniform interoperability layer between models and tools, enabling a true "plug-and-play" ecosystem.

### Standardization: The Hardware Analogy

To understand MCP, imagine the connectivity between a computer and a mouse. A peripheral manufacturer does not construct a mouse solely for a specific computer; instead, they adhere to the **USB standard**. This ensures that any computer equipped with a USB port can use any mouse designed for that standard.

In the MCP Architecture:
- The **MCP Host** is like the computer (e.g., Claude Desktop, an IDE, or an AI Agent).
- The **MCP Server** is like the peripheral (e.g., Google Drive, a database server).
- The **Protocol** corresponds to the USB communication standard.

By this logic, a developer creates a "Google Drive MCP Server" **once**, and that single implementation can be plugged into any Host, effectively collapsing the $N \times M$ complexity into $1 \times M$.

The primary tenets of MCP are:
- **Model Agnosticism**: The protocol is indifferent to which model weights, architecture, or provider is used.
- **Client-Controlled Security**: Host-side controls create a boundary of trust; the agent's capabilities are not left unchecked, as the host acts as a gatekeeper.
- **Composability**: A single host can connect to multiple servers simultaneously, allowing for complex emergent behaviors.

### Architectural Anatomy

**1. MCP Host**
The Host is the application layer where the user interacts with the AI. It is responsible for the user interface, process management, and context orchestration. It manages lifecycle operations, launches MCP servers as local subprocesses, and maintains connections (often via stdio or HTTP). The Host determines when to send information from a server to an LLM.

**2. MCP Client**
The Client is the internal protocol implementation inside the Host. There is a 1:1 connection with the MCP Server. Its primary responsibilities are message framing and state management. It handles JSON-RPC serialization and manages connections/route requests (like `tools/call`). Clients isolate different servers from each other; only the Host is allowed to aggregate context. This security boundary ensures the implementation serves the user's intent and prevents cross-contamination.

**3. MCP Server**
The Server is the bridge to the external world. It is a lightweight, specialized application that wraps a specific data source or toolset and translates its native API into the MCP standard. Servers are defined by their **Capabilities**. During the initiation handshake, the server declares what primitives it supports so the host can adapt its UI and logic. While the connection is stateful, MCP servers are typically designed to be stateless in operation.

### The Core Primitives

Functional MCP consists of three distinct modalities through which an AI model can interact with the external environment:

**Resources: Passive Context (Reading)**
Resources represent data that the model can "read." Think of them as 'books', 'documents', or 'files'. An assistant can open them to answer a query but cannot rewrite the pages. A Resource is any entity identifiable by a URI (Uniform Resource Identifier).

*Example*: Consider an MCP server for an internal documentation platform. It might expose a resource `docs://api/v1/endpoints`. When a user asks, "How do I use the login endpoint?", the Host recognizes the relevance, reads this resource, and provides the content to the LLM. The model then generates an answer based on the exact documentation rather than its training memory.

The Resource primitive also supports a **subscription model**. A client can subscribe to a specific URI; if the underlying data changes (e.g., a new log entry), the server sends a `notifications/resources/updated` message.

**Tools: The Active Agency (Doing)**
Tools are executable functions. They allow AI to actively manipulate the environment: writing a file, sending an email, executing a SQL query, or deploying code. The server lists tools using a **strictly typed JSON schema**. The LLM calls these tools by generating a structured request which the Host executes. 

Strict typing is essential for reliability. Furthermore, a **Human-in-the-Loop (HITL)** approval process is often employed to prevent unauthorized autonomous actions—a critical security control.

**Prompts: The Standardized Interface (Guiding)**
Prompts are reusable templates for interaction. Returning to the library analogy, Prompts are standardized forms or "request slips" available at the help desk. They help users or the AI structure their intent effectively to achieve specific outcomes.

### Protocol Mechanics: The Data Layer

MCP is built on top of **JSON-RPC 2.0**, a stateless, lightweight remote procedure call protocol. This choice ensures broad compatibility and easy parsing.

**Example Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": { "name": "get_weather", "arguments": {"city": "Paris"} }
}
```

**Example Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { "temperature": "15°C", "condition": "Rain" }
}
```

**Transport Layer:**
MCP defines two primary transport mechanisms:
1.  **Stdio Transport**: The default for local development and desktop apps (like Claude Desktop). Data flows over Standard Input (stdin) and Standard Output (stdout).
2.  **HTTP/SSE Transport**: Used when the server runs on a remote machine. It employs a dual-channel approach: **Server-Sent Events (SSE)** for server-to-client messages (push), and standard **HTTP POST** requests for client-to-server messages.

### Practical Example: Python SDK

Utilizing the **FastMCP** abstraction allows for rapid development:

```python
from mcp.server.fastmcp import FastMCP

# Instantiate the server
mcp = FastMCP("Weather Service")

# Define a tool using the @mcp.tool decorator
@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """
    Get weather forecast for a location.
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # Logic to call external Weather API
    return "Sunny, 25C"

# Define a resource using a URI template
@mcp.resource("weather://{state}/alerts")
async def get_alerts(state: str) -> str:
    """Get active alerts for a state"""
    return "Flood Warning: Moderate"
```

### Advanced Capabilities: Sampling and Pagination

The most futuristic aspect of MCP is **Sampling**. This feature flips the control flow: The Server asks the Host (and thus the LLM) to generate content. This allows the Server to leverage the Host's intelligence for tasks like semantic description of raw binary data.

When dealing with massive resources (e.g., a log file with 1 million lines), MCP supports **Pagination**. The Client requests resources, the Server returns a batch with a `nextCursor`, and the Client follows up as needed. This mechanism is critical for performance and cost management in token-based billing models.

### MCP vs. Language Server Protocol (LSP)

The strongest conceptual parallel to MCP is the **Language Server Protocol (LSP)**. LSP standardized how IDEs (VS Code, Vim) talk to language compilers.

- **LSP**: Client = IDE, Server = Language Analyzer. Goal = IDE features (Autocomplete, Go to Definition).
- **MCP**: Client = LLM Host, Server = Context Provider. Goal = Retrieval Augmented Generation (RAG) and Action execution.

**Insight**: MCP is essentially "**LSP for AI**." Just as LSP stopped every text editor from writing its own bespoke Java parser, MCP stops every AI application from writing its own bespoke Jira integration.

### Why not just use REST APIs?

A common question is: *"Why not just let LLMs call REST APIs directly?"*

1.  **Context Window Limits**: APIs often have thousands of endpoints with verbose documentation. Dumping a full Swagger definition into an LLM context is expensive and error-prone.
2.  **Abstraction**: MCP Servers act as a "Curated API," exposing only high-level, relevant tools.
3.  **State Management**: MCP manages connection states and resource subscriptions, which stateless REST APIs do not natively handle. This enables persistent, real-time monitoring.



