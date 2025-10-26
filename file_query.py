#!/usr/bin/env python3
"""Quick start example for Claude Code SDK."""

import time
import anyio
import click
import json
import os

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

async def process_questions_json(input_file, questions_file):
    """Process a JSON file with questions and generate results with Claude answers."""
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    results = []
    
    for i, item in enumerate(questions_data):
        print(f"\n{'='*60}")
        print(f"Processing question {i+1}/{len(questions_data)}")
        print(f"Question: {item['question']}")
        print(f"{'='*60}")
        
        # Get Claude's answer
        claude_answer = await get_claude_answer(input_file, item['question'])
        
        # Create result entry
        result_item = {
            "question": item['question'],
            "expected_answer": item['answer'],
            "claude_answer": claude_answer
        }
        results.append(result_item)
    
    # Write results to output file
    output_file = questions_file.replace('.json', '_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

async def get_claude_answer(filename, query_text):
    """Get Claude's answer for a single question."""
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Grep", "Glob"],
        system_prompt="You are a helpful file assistant. You must use tools to answer the question.",
        model="sonnet",
    )

    prompt = f"Using file {filename} as a source, answer the question: {query_text}"

    tools_used = False
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
                    print(f"🔧 Tool Use: {block.name}")
                elif isinstance(block, ToolResultBlock):
                    if block.is_error:
                        print(f"❌ Error: {block.content}")
            
            # Only keep text from the last message (final answer)
            if current_message_text:
                last_message_text = current_message_text
                
        elif isinstance(message, ResultMessage) and message.total_cost_usd > 0:
            print(f"Cost: ${message.total_cost_usd:.4f}")
    
    # Return the final answer
    return ' '.join(last_message_text) if last_message_text else "No answer generated"

@click.command()
@click.option("-i", "--input-file", required=True, help="JSON file containing questions and expected answers")
@click.option("-q", "--questions-file", required=True, help="Source file to query for answers")
def main(input_file, questions_file):
    """Process a JSON file with questions and generate results with Claude answers."""
    async def run_processing():
        time_start = time.time()
        await process_questions_json(input_file, questions_file)
        time_end = time.time()
        print(f"\nTotal time taken: {time_end - time_start:.2f} seconds")
    
    anyio.run(run_processing)


if __name__ == "__main__":
    main()
