"""
Example Usage Script

Demonstrates how to use the Black Swan Scenario Analysis System
programmatically for custom workflows.
"""

import json
from pathlib import Path


def example_basic_usage():
    """Basic example: Generate scenarios and evaluate strategies."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    from scenario_generator import ScenarioGenerator
    from nlp_translator import ScenarioTranslator
    from fuzzy_mcdm import StrategyEvaluator
    
    # Step 1: Generate scenarios
    generator = ScenarioGenerator(llm_provider="mock", output_dir="data/examples")
    scenarios = generator.generate_scenarios(num_scenarios=50, batch_size=10)
    print(f"Generated {len(scenarios)} scenarios\n")
    
    # Step 2: Translate to quantitative parameters
    translator = ScenarioTranslator()
    quantitative = translator.translate_scenarios_batch(scenarios)
    print(f"Translated {len(quantitative)} scenarios\n")
    
    # Step 3: Evaluate strategies
    evaluator = StrategyEvaluator(mcdm_method="pythagorean_topsis")
    results = evaluator.evaluate_strategies(quantitative)
    
    # Print top 3 strategies
    print("Top 3 Resilience Strategies:")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result['strategy']} (Score: {result['score']:.4f})")
    print()


def example_custom_strategies():
    """Example with custom strategies and criteria."""
    print("=" * 60)
    print("EXAMPLE 2: Custom Strategies and Criteria")
    print("=" * 60)
    
    from fuzzy_mcdm import StrategyEvaluator
    
    # Define your own strategies
    custom_strategies = [
        "Blockchain Supply Chain",
        "AI-Powered Demand Forecasting",
        "Circular Economy Model",
        "Regional Manufacturing Hubs"
    ]
    
    # Define custom criteria
    custom_criteria = [
        "Sustainability Impact",
        "Technology Readiness",
        "Investment Required",
        "Time to Implement"
    ]
    
    # Load existing quantitative scenarios
    with open("data/output/scenarios_quantitative.json", 'r') as f:
        scenarios = json.load(f)
    
    # Evaluate with custom settings
    evaluator = StrategyEvaluator(mcdm_method="q_rung_vikor")
    results = evaluator.evaluate_strategies(
        scenarios,
        strategies=custom_strategies,
        criteria=custom_criteria
    )
    
    print("Custom Strategy Rankings:")
    for result in results:
        print(f"  {result['rank']}. {result['strategy']} (Q={result['Q']:.4f})")
    print()


def example_compare_methods():
    """Compare different MCDM methods."""
    print("=" * 60)
    print("EXAMPLE 3: Compare MCDM Methods")
    print("=" * 60)
    
    from fuzzy_mcdm import StrategyEvaluator
    
    # Load scenarios
    with open("data/output/scenarios_quantitative.json", 'r') as f:
        scenarios = json.load(f)
    
    methods = ["pythagorean_topsis", "q_rung_vikor"]
    
    for method in methods:
        evaluator = StrategyEvaluator(mcdm_method=method)
        results = evaluator.evaluate_strategies(scenarios)
        
        print(f"\n{method.upper()}:")
        print(f"  Best: {results[0]['strategy']}")
        print(f"  Worst: {results[-1]['strategy']}")
    
    print()


def example_load_and_analyze():
    """Load existing scenarios and perform analysis."""
    print("=" * 60)
    print("EXAMPLE 4: Load and Analyze Existing Scenarios")
    print("=" * 60)
    
    from nlp_translator import ScenarioTranslator
    from fuzzy_mcdm import StrategyEvaluator
    
    # Load raw scenarios
    with open("data/output/scenarios_raw.json", 'r') as f:
        raw_scenarios = json.load(f)
    
    print(f"Loaded {len(raw_scenarios)} raw scenarios")
    
    # Analyze scenario characteristics
    severities = {}
    for s in raw_scenarios:
        sev = s.get('severity', 'Unknown')
        severities[sev] = severities.get(sev, 0) + 1
    
    print("\nScenario Severity Distribution:")
    for severity, count in sorted(severities.items()):
        print(f"  {severity}: {count}")
    
    # Translate and evaluate
    translator = ScenarioTranslator()
    quantitative = translator.translate_scenarios_batch(raw_scenarios)
    
    evaluator = StrategyEvaluator(mcdm_method="pythagorean_topsis")
    results = evaluator.evaluate_strategies(quantitative)
    
    # Generate detailed report
    report = evaluator.generate_report(results)
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_custom_strategies()
    example_compare_methods()
    example_load_and_analyze()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
