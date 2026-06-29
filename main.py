"""
Main Entry Point

GenAI-Driven Black Swan Scenario Generation + Fuzzy MCDM System

This script orchestrates the complete workflow:
1. Generate unprecedented supply chain disruption scenarios using LLMs
2. Translate qualitative scenarios to quantitative parameters using NLP
3. Evaluate resilience strategies using advanced Fuzzy MCDM methods
"""

import argparse
import json
from pathlib import Path
from loguru import logger
import sys


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    logger.remove()  # Remove default handler
    
    if verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    logger.add(
        "logs/black_swan_system.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )


def run_generation(num_scenarios: int, output_dir: str, llm_provider: str):
    """Run scenario generation phase."""
    from scenario_generator import ScenarioGenerator
    
    logger.info(f"Generating {num_scenarios} Black Swan scenarios...")
    
    generator = ScenarioGenerator(
        llm_provider=llm_provider,
        output_dir=output_dir
    )
    
    scenarios = generator.generate_scenarios(
        num_scenarios=num_scenarios,
        batch_size=10,
        refine=True
    )
    
    logger.success(f"Generated {len(scenarios)} scenarios")
    return scenarios


def run_translation(scenarios_file: str, output_file: str):
    """Run NLP translation phase."""
    from nlp_translator import ScenarioTranslator
    
    logger.info(f"Loading scenarios from {scenarios_file}...")
    
    with open(scenarios_file, 'r') as f:
        scenarios = json.load(f)
    
    translator = ScenarioTranslator()
    quantitative = translator.translate_scenarios_batch(scenarios)
    
    # Save quantitative results
    translator.export_to_csv(quantitative, output_file)
    
    # Also save as JSON
    json_path = output_file.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump(quantitative, f, indent=2)
    
    logger.success(f"Translated {len(quantitative)} scenarios to {output_file}")
    return quantitative


def run_evaluation(scenarios_file: str, method: str, output_report: str):
    """Run strategy evaluation phase."""
    from fuzzy_mcdm import StrategyEvaluator
    
    logger.info(f"Loading quantitative scenarios from {scenarios_file}...")
    
    with open(scenarios_file, 'r') as f:
        scenarios = json.load(f)
    
    logger.info(f"Evaluating strategies using {method}...")
    
    evaluator = StrategyEvaluator(mcdm_method=method)
    results = evaluator.evaluate_strategies(scenarios)
    
    # Generate report
    evaluator.generate_report(results, output_report)
    
    return results


def run_full_pipeline(
    num_scenarios: int,
    mcdm_method: str,
    llm_provider: str,
    output_dir: str
):
    """Run the complete pipeline from generation to evaluation."""
    logger.info("Starting full Black Swan analysis pipeline...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Generate scenarios
    logger.info("=" * 60)
    logger.info("PHASE 1: Scenario Generation")
    logger.info("=" * 60)
    
    scenarios_file = output_path / "scenarios_raw.json"
    scenarios = run_generation(num_scenarios, str(output_path), llm_provider)
    
    # Save raw scenarios to expected file path
    raw_scenarios_path = output_path / "scenarios_raw.json"
    with open(raw_scenarios_path, 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    # Phase 2: Translate to quantitative
    logger.info("=" * 60)
    logger.info("PHASE 2: NLP Translation")
    logger.info("=" * 60)
    
    quantitative_file = output_path / "scenarios_quantitative.csv"
    quantitative = run_translation(str(scenarios_file), str(quantitative_file))
    
    # Phase 3: Evaluate strategies
    logger.info("=" * 60)
    logger.info("PHASE 3: Strategy Evaluation")
    logger.info("=" * 60)
    
    report_file = output_path / "strategy_ranking_report.txt"
    results = run_evaluation(
        str(quantitative_file).replace('.csv', '.json'),
        mcdm_method,
        str(report_file)
    )
    
    logger.info("=" * 60)
    logger.success("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Report: {report_file}")
    
    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="GenAI-Driven Black Swan Scenario Generation + Fuzzy MCDM System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with 1000 scenarios
  python main.py --mode full --num-scenarios 1000
  
  # Only generate scenarios
  python main.py --mode generate --num-scenarios 500
  
  # Evaluate with VIKOR method
  python main.py --mode evaluate --method q_rung_vikor
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'generate', 'translate', 'evaluate'],
        default='full',
        help='Operation mode (default: full)'
    )
    
    parser.add_argument(
        '--num-scenarios',
        type=int,
        default=100,
        help='Number of scenarios to generate (default: 100)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['pythagorean_topsis', 'q_rung_vikor'],
        default='pythagorean_topsis',
        help='MCDM method for evaluation (default: pythagorean_topsis)'
    )
    
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['openai', 'huggingface', 'mock'],
        default='mock',
        help='LLM provider for scenario generation (default: mock)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='Output directory for results (default: data/output)'
    )
    
    parser.add_argument(
        '--scenarios-file',
        type=str,
        help='Path to existing scenarios file (for translate/evaluate modes)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose/debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger.info("Black Swan Scenario Analysis System starting...")
    logger.info(f"Mode: {args.mode}, Method: {args.method}")
    
    try:
        if args.mode == 'full':
            results = run_full_pipeline(
                num_scenarios=args.num_scenarios,
                mcdm_method=args.method,
                llm_provider=args.llm_provider,
                output_dir=args.output_dir
            )
        
        elif args.mode == 'generate':
            results = run_generation(
                num_scenarios=args.num_scenarios,
                output_dir=args.output_dir,
                llm_provider=args.llm_provider
            )
        
        elif args.mode == 'translate':
            if not args.scenarios_file:
                logger.error("--scenarios-file required for translate mode")
                sys.exit(1)
            
            output_file = Path(args.output_dir) / "scenarios_quantitative.csv"
            results = run_translation(args.scenarios_file, str(output_file))
        
        elif args.mode == 'evaluate':
            if not args.scenarios_file:
                logger.error("--scenarios-file required for evaluate mode")
                sys.exit(1)
            
            report_file = Path(args.output_dir) / "strategy_report.txt"
            results = run_evaluation(
                args.scenarios_file,
                args.method,
                str(report_file)
            )
        
        logger.success("Operation completed successfully!")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
