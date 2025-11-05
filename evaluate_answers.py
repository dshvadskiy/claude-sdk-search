#!/usr/bin/env python3
"""
Script to evaluate Claude answers using deepeval framework with LLM as judge.
Evaluates answers against expected answers and calculates scores.
"""

import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
import click
from pydantic import BaseModel, Field

# Remove deepeval imports for now and use direct OpenAI evaluation
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EvaluationResult(BaseModel):
    """Structured output for evaluation results."""
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Accuracy score from 0.0 to 1.0")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Completeness score from 0.0 to 1.0")
    reasoning: str = Field(..., description="Detailed reasoning for the evaluation")


@dataclass
class QAResult:
    """Data class for question-answer evaluation results."""
    question: str
    expected_answer: str
    claude_answer: str
    accuracy_score: float
    completeness_score: float
    reasoning: str


class ClaudeAnswerEvaluator:
    """Evaluator for Claude answers using deepeval with LLM judge."""
    
    def __init__(self, api_key: str = None):
        """Initialize the evaluator with API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # No need for deepeval metrics, we'll use direct OpenAI evaluation
    
    def load_qa_data(self, json_file: str) -> List[Dict[str, Any]]:
        """Load question-answer data from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_single_answer(self, question: str, claude_answer: str, expected_answer: str) -> EvaluationResult:
        """Evaluate a single answer using OpenAI with structured output."""
        prompt = f"""
You are an expert evaluator. Please evaluate the quality of the following answer to a question.

Question: {question}

Answer: {claude_answer}

Expected Answer: {expected_answer}

Please evaluate the answer on the following criteria and provide detailed scores:
1. Accuracy: Is the answer factually correct? (0.0-1.0)
2. Completeness: Does the answer fully address the question? (0.0-1.0)

Provide a brief reasoning for your scores.
"""

        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=EvaluationResult,
                # temperature=0.1,
                # max_tokens=1000
            )
            
            return response.choices[0].message.parsed
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            # Return a default evaluation result on error
            return EvaluationResult(
                accuracy_score=0.0,
                completeness_score=0.0,
                reasoning=f"Error during evaluation: {e}"
            )
    
    def evaluate_answers(self, json_file: str) -> tuple[List[QAResult], List[EvaluationResult]]:
        """Evaluate all answers in the JSON file."""
        print(f"Loading data from {json_file}...")
        qa_data = self.load_qa_data(json_file)
        
        print(f"Evaluating {len(qa_data)} questions...")
        
        # Evaluate each answer individually
        results = []
        evaluations = []
        for i, item in enumerate(qa_data):
            print(f"Evaluating question {i+1}/{len(qa_data)}...")
            question = item["question"]
            claude_answer = item["claude_answer"]
            expected_answer = item["expected_answer"]
            
            evaluation = self.evaluate_single_answer(question, claude_answer, expected_answer)
            evaluations.append(evaluation)
            
            result = QAResult(
                question=question,
                expected_answer=expected_answer,
                claude_answer=claude_answer,
                accuracy_score=evaluation.accuracy_score,  # Use accuracy score as the main score
                completeness_score=evaluation.completeness_score,
                reasoning=evaluation.reasoning
            )
            results.append(result)
        
        return results, evaluations
    
    def calculate_aggregate_scores(self, results: List[QAResult]) -> Dict[str, float]:
        """Calculate aggregate scores from individual results."""
        if not results:
            return {}
        
        accuracy_scores = [r.accuracy_score for r in results]
        completeness_scores = [r.completeness_score for r in results]
        
        return {
            "average_accuracy_score": sum(accuracy_scores) / len(accuracy_scores),
            "min_accuracy_score": min(accuracy_scores),
            "max_accuracy_score": max(accuracy_scores),
            "average_completeness_score": sum(completeness_scores) / len(completeness_scores),
            "min_completeness_score": min(completeness_scores),
            "max_completeness_score": max(completeness_scores),
            "total_questions": len(results),
        }
    
    def save_detailed_results(self, evaluations: List[EvaluationResult], qa_data: List[Dict[str, Any]], output_file: str):
        """Save detailed evaluation results with structured data to JSON file."""
        detailed_results = []
        for i, (evaluation, item) in enumerate(zip(evaluations, qa_data)):
            detailed_results.append({
                "question": item["question"],
                "claude_answer": item["claude_answer"],
                "expected_answer": item["expected_answer"],
                "evaluation": {
                    "accuracy_score": evaluation.accuracy_score,
                    "completeness_score": evaluation.completeness_score,
                    "reasoning": evaluation.reasoning
                }
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to {output_file}")


@click.command()
@click.option('--input-file', '-i', 
              required=True,
              help='Input JSON file with question-answer data')
@click.option('--output-file', '-o',
              help='Output file for evaluation results (auto-generated if not specified)')
@click.option('--api-key', '-k',
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
def main(input_file: str, output_file: str, api_key: str):
    """Evaluate Claude answers using deepeval with LLM judge."""
    
    try:
        # Generate output file name if not provided
        if not output_file:
            from pathlib import Path
            input_path = Path(input_file)
            base_name = input_path.stem
            output_dir = input_path.parent
            output_file = str(output_dir / f"{base_name}_basic_evaluation.json")
        
        # Initialize evaluator
        evaluator = ClaudeAnswerEvaluator(api_key=api_key)
        
        # Run evaluation
        results, evaluations = evaluator.evaluate_answers(input_file)
        
        # Calculate aggregate scores
        aggregate_scores = evaluator.calculate_aggregate_scores(results)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {aggregate_scores['total_questions']}")
        print(f"Average Accuracy Score: {aggregate_scores['average_accuracy_score']:.3f}")
        print(f"Average Completeness Score: {aggregate_scores['average_completeness_score']:.3f}")
        print(f"Min Accuracy Score: {aggregate_scores['min_accuracy_score']:.3f}")
        print(f"Max Accuracy Score: {aggregate_scores['max_accuracy_score']:.3f}")
        print(f"Min Completeness Score: {aggregate_scores['min_completeness_score']:.3f}")
        print(f"Max Completeness Score: {aggregate_scores['max_completeness_score']:.3f}")
        # print(f"Min Score: {aggregate_scores['min_score']:.3f}")
        # print(f"Max Score: {aggregate_scores['max_score']:.3f}")
        # print(f"High Quality (≥0.8): {aggregate_scores['high_quality_answers']}")
        # print(f"Medium Quality (0.6-0.8): {aggregate_scores['medium_quality_answers']}")
        # print(f"Low Quality (<0.6): {aggregate_scores['low_quality_answers']}")
        print("="*60)
        
        # Print individual results with detailed breakdown
        print("\nINDIVIDUAL RESULTS:")
        print("-" * 60)
        for i, (result, evaluation) in enumerate(zip(results, evaluations), 1):
            print(f"\nQuestion {i}:")
            print(f"Accuracy: {evaluation.accuracy_score:.3f}")
            print(f"Completeness: {evaluation.completeness_score:.3f}")
            print(f"Question: {result.question[:100]}...")
            print(f"Reasoning: {result.reasoning}")
        
        
        # Save detailed results
        detailed_output_file = output_file.replace('.json', '_detailed.json')
        evaluator.save_detailed_results(evaluations, evaluator.load_qa_data(input_file), detailed_output_file)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
