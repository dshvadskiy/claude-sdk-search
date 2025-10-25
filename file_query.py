#!/usr/bin/env python3
"""Quick start example for Claude Code SDK."""

import time
import anyio
import click

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    query,
)

async def with_tools_example(filename, query_text):
    """Example using file tools."""
    # print("=== With File Tools Example ===")

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Grep", "Glob"],
        system_prompt="You are a helpful file assistant. You must use tools to answer the question.",
        model="sonnet",
    )

    prompt = f"Using file {filename} as a source, answer the question: {query_text}"

    tools_used = False
    final_text_blocks = []
    last_message_text = []
    
    async for message in query(
        prompt=prompt,
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            current_message_text = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    current_message_text.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tools_used = True
                    print(f"\n🔧 Tool Use: {block.name}")
                    print(f"   Input: {block.input}")
                elif isinstance(block, ToolResultBlock):
                    print(f"   Result: {block.content}")
                    if block.is_error:
                        print(f"   ❌ Error: {block.content}")
            
            # Only keep text from the last message (final answer)
            if current_message_text:
                last_message_text = current_message_text
                
        elif isinstance(message, ResultMessage) and message.total_cost_usd > 0:
            print(f"\nCost: ${message.total_cost_usd:.4f}")
            print(f"\nUsage: {message.usage}")
    
    # Only show text from the very last message (final answer)
    if last_message_text:
        print(f"\nClaude: {' '.join(last_message_text)}")
    print()

@click.command()
@click.argument("filename", metavar="<filename>")
@click.argument("query", metavar="<query>")
def main(filename, query):
    """Query a file using Claude Code SDK.
    
    FILENAME: Name of the file to query
    QUERY: Question to ask about the file
    """
    async def run_query():
        time_start = time.time()
        await with_tools_example(filename, query)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start:.2f} seconds")
    
    anyio.run(run_query)


if __name__ == "__main__":
    main()
