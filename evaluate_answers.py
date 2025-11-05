#!/usr/bin/env python3
"""
Script to evaluate Claude answers using deepeval framework with LLM as judge.
Evaluates answers against expected answers and calculates scores.
"""

import json
import os
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import click
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Remove deepeval imports for now and use direct OpenAI evaluation
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    def __init__(self, api_key: str = None, model: str = None, max_workers: int = 5):
        """Initialize the evaluator with API key.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for evaluation (defaults to OPENAI_MODEL env var or gpt-4o-mini)
            max_workers: Maximum number of concurrent evaluation threads (default: 5)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_workers = max_workers
        self.openai_client = OpenAI(api_key=self.api_key)

        logger.info(f"Initialized evaluator with model: {self.model}, max_workers: {self.max_workers}")

        # No need for deepeval metrics, we'll use direct OpenAI evaluation
    
    def load_qa_data(self, json_file: str) -> List[Dict[str, Any]]:
        """Load question-answer data from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        reraise=True
    )
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
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=EvaluationResult,
                temperature=0.1,
            )

            return response.choices[0].message.parsed

        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            # Return a default evaluation result on error
            return EvaluationResult(
                accuracy_score=0.0,
                completeness_score=0.0,
                reasoning=f"Error during evaluation: {e}"
            )
    
    def evaluate_answers(self, json_file: str, use_concurrent: bool = True) -> Tuple[List[QAResult], List[EvaluationResult]]:
        """Evaluate all answers in the JSON file.

        Args:
            json_file: Path to JSON file containing questions and answers
            use_concurrent: Whether to use concurrent processing (default: True)

        Returns:
            Tuple of (results list, evaluations list)
        """
        logger.info(f"Loading data from {json_file}...")
        qa_data = self.load_qa_data(json_file)

        logger.info(f"Evaluating {len(qa_data)} questions using {self.model}...")

        if use_concurrent and len(qa_data) > 1:
            return self._evaluate_concurrent(qa_data)
        else:
            return self._evaluate_sequential(qa_data)

    def _evaluate_sequential(self, qa_data: List[Dict[str, Any]]) -> Tuple[List[QAResult], List[EvaluationResult]]:
        """Evaluate answers sequentially with progress bar."""
        results = []
        evaluations = []

        for item in tqdm(qa_data, desc="Evaluating questions", unit="question"):
            question = item["question"]
            claude_answer = item["claude_answer"]
            expected_answer = item["expected_answer"]

            evaluation = self.evaluate_single_answer(question, claude_answer, expected_answer)
            evaluations.append(evaluation)

            result = QAResult(
                question=question,
                expected_answer=expected_answer,
                claude_answer=claude_answer,
                accuracy_score=evaluation.accuracy_score,
                completeness_score=evaluation.completeness_score,
                reasoning=evaluation.reasoning
            )
            results.append(result)

        return results, evaluations

    def _evaluate_concurrent(self, qa_data: List[Dict[str, Any]]) -> Tuple[List[QAResult], List[EvaluationResult]]:
        """Evaluate answers concurrently with progress bar."""
        results = [None] * len(qa_data)
        evaluations = [None] * len(qa_data)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, item in enumerate(qa_data):
                future = executor.submit(
                    self.evaluate_single_answer,
                    item["question"],
                    item["claude_answer"],
                    item["expected_answer"]
                )
                future_to_index[future] = (i, item)

            # Process completed tasks with progress bar
            with tqdm(total=len(qa_data), desc="Evaluating questions", unit="question") as pbar:
                for future in as_completed(future_to_index):
                    i, item = future_to_index[future]
                    try:
                        evaluation = future.result()
                        evaluations[i] = evaluation

                        result = QAResult(
                            question=item["question"],
                            expected_answer=item["expected_answer"],
                            claude_answer=item["claude_answer"],
                            accuracy_score=evaluation.accuracy_score,
                            completeness_score=evaluation.completeness_score,
                            reasoning=evaluation.reasoning
                        )
                        results[i] = result
                    except Exception as e:
                        logger.error(f"Failed to evaluate question {i+1}: {e}")
                        # Create a failed result
                        evaluations[i] = EvaluationResult(
                            accuracy_score=0.0,
                            completeness_score=0.0,
                            reasoning=f"Evaluation failed: {e}"
                        )
                        results[i] = QAResult(
                            question=item["question"],
                            expected_answer=item["expected_answer"],
                            claude_answer=item["claude_answer"],
                            accuracy_score=0.0,
                            completeness_score=0.0,
                            reasoning=f"Evaluation failed: {e}"
                        )
                    finally:
                        pbar.update(1)

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

        logger.info(f"Detailed results saved to {output_file}")


@click.command()
@click.option('--input-file', '-i',
              required=True,
              help='Input JSON file with question-answer data')
@click.option('--output-file', '-o',
              help='Output file for evaluation results (auto-generated if not specified)')
@click.option('--api-key', '-k',
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model', '-m',
              help='OpenAI model to use for evaluation (default: gpt-4o-mini or OPENAI_MODEL env var)')
@click.option('--max-workers', '-w',
              type=int,
              default=5,
              help='Maximum number of concurrent evaluation threads (default: 5)')
@click.option('--no-concurrent',
              is_flag=True,
              help='Disable concurrent processing and evaluate sequentially')
def main(input_file: str, output_file: str, api_key: str, model: str, max_workers: int, no_concurrent: bool):
    """Evaluate Claude answers using OpenAI with LLM as judge."""

    try:
        # Generate output file name if not provided
        if not output_file:
            from pathlib import Path
            input_path = Path(input_file)
            base_name = input_path.stem
            output_dir = input_path.parent
            output_file = str(output_dir / f"{base_name}_basic_evaluation.json")

        # Initialize evaluator
        evaluator = ClaudeAnswerEvaluator(api_key=api_key, model=model, max_workers=max_workers)
        
        # Run evaluation
        results, evaluations = evaluator.evaluate_answers(input_file, use_concurrent=not no_concurrent)

        # Calculate aggregate scores
        aggregate_scores = evaluator.calculate_aggregate_scores(results)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Questions: {aggregate_scores['total_questions']}")
        logger.info(f"Average Accuracy Score: {aggregate_scores['average_accuracy_score']:.3f}")
        logger.info(f"Average Completeness Score: {aggregate_scores['average_completeness_score']:.3f}")
        logger.info(f"Min Accuracy Score: {aggregate_scores['min_accuracy_score']:.3f}")
        logger.info(f"Max Accuracy Score: {aggregate_scores['max_accuracy_score']:.3f}")
        logger.info(f"Min Completeness Score: {aggregate_scores['min_completeness_score']:.3f}")
        logger.info(f"Max Completeness Score: {aggregate_scores['max_completeness_score']:.3f}")
        logger.info("="*60)

        # Print individual results with detailed breakdown
        logger.info("\nINDIVIDUAL RESULTS:")
        logger.info("-" * 60)
        for i, (result, evaluation) in enumerate(zip(results, evaluations), 1):
            logger.info(f"\nQuestion {i}:")
            logger.info(f"Accuracy: {evaluation.accuracy_score:.3f}")
            logger.info(f"Completeness: {evaluation.completeness_score:.3f}")
            logger.info(f"Question: {result.question[:100]}...")
            logger.info(f"Reasoning: {result.reasoning}")


        # Save detailed results
        detailed_output_file = output_file.replace('.json', '_detailed.json')
        evaluator.save_detailed_results(evaluations, evaluator.load_qa_data(input_file), detailed_output_file)

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
