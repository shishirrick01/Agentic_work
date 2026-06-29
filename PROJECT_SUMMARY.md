# GenAI-Driven Black Swan Scenario Analysis System

## 🎯 Project Overview

This system implements an innovative approach to supply chain resilience planning by combining:

1. **Generative AI** for unprecedented "Black Swan" scenario generation
2. **NLP Translation** to convert qualitative scenarios into quantitative parameters  
3. **Advanced Fuzzy MCDM** (Pythagorean Fuzzy TOPSIS & q-Rung Orthopair VIKOR) for strategy evaluation

## ✨ Key Features

### 1. LLM-Powered Scenario Generation
- Generates 1,000+ unique Black Swan scenarios
- Multi-node supply chain disruptions
- Compound events with cascading effects
- Focus areas: climate, geopolitical, technological, general
- Supports OpenAI, HuggingFace, or mock LLM providers

### 2. NLP Translation Layer
- Converts text descriptions to quantitative metrics:
  - Lead time increases (%)
  - Cost increases (%)
  - Capacity reductions (%)
  - Reliability scores
  - Recovery times
- Keyword-based impact extraction
- Severity and duration adjustments

### 3. Fuzzy MCDM Evaluation
- **Pythagorean Fuzzy TOPSIS**: Handles uncertainty where μ² + ν² ≤ 1
- **q-Rung Orthopair VIKOR**: Generalized fuzzy sets (μ^q + ν^q ≤ 1)
- Evaluates strategies across all generated scenarios
- Produces ranked recommendations

## 📁 Project Structure

```
/workspace/
├── main.py                      # Main CLI entry point
├── examples.py                  # Usage examples
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── scenario_generator/          # LLM scenario generation
│   ├── __init__.py
│   ├── generator.py            # Core generation logic
│   └── prompt_templates.py     # Prompt engineering templates
│
├── nlp_translator/             # Qualitative → Quantitative
│   ├── __init__.py
│   ├── translator.py           # Main translation logic
│   └── parameter_extractor.py  # NLP parameter extraction
│
├── fuzzy_mcdm/                 # Multi-Criteria Decision Making
│   ├── __init__.py
│   ├── pythagorean_fuzzy_topsis.py    # PF-TOPSIS implementation
│   ├── q_rung_orthopair_vikor.py      # q-ROF VIKOR implementation
│   └── strategy_evaluator.py   # Integrated evaluation
│
├── config/                     # Configuration files
│   └── default_config.yaml
│
├── data/                       # Generated data
│   ├── output/                 # Pipeline outputs
│   └── scenarios/              # Raw scenarios
│
└── utils/                      # Utility functions
    └── __init__.py
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Run Full Pipeline (Generate → Translate → Evaluate)
```bash
python main.py --mode full --num-scenarios 1000 --method pythagorean_topsis
```

#### Generate Scenarios Only
```bash
python main.py --mode generate --num-scenarios 500 --llm-provider mock
```

#### Evaluate with VIKOR Method
```bash
python main.py --mode evaluate --scenarios-file data/output/scenarios_quantitative.json --method q_rung_vikor
```

#### Use OpenAI for Real Scenario Generation
```bash
export OPENAI_API_KEY="your-key-here"
python main.py --mode full --num-scenarios 100 --llm-provider openai
```

### Programmatic Usage

```python
from scenario_generator import ScenarioGenerator
from nlp_translator import ScenarioTranslator
from fuzzy_mcdm import StrategyEvaluator

# Step 1: Generate scenarios
generator = ScenarioGenerator(llm_provider="mock")
scenarios = generator.generate_scenarios(num_scenarios=100)

# Step 2: Translate to quantitative
translator = ScenarioTranslator()
quantitative = translator.translate_scenarios_batch(scenarios)

# Step 3: Evaluate strategies
evaluator = StrategyEvaluator(mcdm_method="pythagorean_topsis")
results = evaluator.evaluate_strategies(quantitative)

# Print top strategies
for r in results[:3]:
    print(f"{r['rank']}. {r['strategy']} (Score: {r['score']:.4f})")
```

## 📊 Default Resilience Strategies Evaluated

1. **Multi-sourcing** - Diversifying supplier base
2. **Safety Stock Holding** - Maintaining buffer inventory
3. **Nearshoring** - Moving production closer to markets
4. **Supplier Diversification** - Reducing single-source dependency
5. **Vertical Integration** - Owning more of supply chain
6. **Digital Twin Implementation** - Virtual supply chain modeling
7. **Agile Manufacturing** - Flexible production systems
8. **Strategic Partnerships** - Long-term collaborative relationships

## 🎓 Academic Contributions

This system addresses critical research gaps:

### Cold Start Problem
Traditional AI models fail at unprecedented events due to lack of training data. Our GenAI approach generates synthetic scenarios without historical bias.

### Uncertainty Modeling
Pythagorean and q-Rung Orthopair fuzzy sets provide superior uncertainty handling compared to traditional crisp or intuitionistic fuzzy methods.

### Synthetic Data for OR
Demonstrates novel application of GenAI for synthetic data generation in Operations Research, a rapidly emerging research area.

## 🔧 Configuration

Edit `config/default_config.yaml` to customize:
- Number of scenarios
- Batch sizes
- LLM provider settings
- MCDM parameters (q value for q-ROF, v for VIKOR)
- Strategy weights
- Output formats

## 📈 Output Files

After running the pipeline:
- `scenarios_raw.json` - Generated Black Swan scenarios
- `scenarios_quantitative.csv/json` - Translated parameters
- `strategy_ranking_report.txt` - Human-readable recommendations

## 🧪 Testing

Run example workflows:
```bash
python examples.py
```

This demonstrates:
- Basic usage workflow
- Custom strategies/criteria
- Method comparison (TOPSIS vs VIKOR)
- Loading and analyzing existing scenarios

## 📝 Example Output

```
============================================================
SUPPLY CHAIN RESILIENCE STRATEGY EVALUATION REPORT
============================================================

MCDM Method: pythagorean_topsis
Number of Strategies Evaluated: 8

------------------------------------------------------------
STRATEGY RANKINGS
------------------------------------------------------------

1. Supplier Diversification       Score: 1.0000
2. Agile Manufacturing            Score: 1.0000
3. Digital Twin Implementation    Score: 0.8013
4. Multi-sourcing                 Score: 0.7708
5. Strategic Partnerships         Score: 0.6649
6. Nearshoring                    Score: 0.6128
7. Safety Stock Holding           Score: 0.5948
8. Vertical Integration           Score: 0.5647

------------------------------------------------------------
RECOMMENDATIONS
------------------------------------------------------------

Top Strategy: Supplier Diversification
```

## 🔬 Technical Details

### Pythagorean Fuzzy Sets
Extend intuitionistic fuzzy sets by allowing:
- μ² + ν² ≤ 1 (instead of μ + ν ≤ 1)
- More flexibility in uncertainty representation
- Better suited for complex decision environments

### q-Rung Orthopair Fuzzy Sets
Generalization where:
- μ^q + ν^q ≤ 1, q ≥ 1
- Higher q values allow more uncertainty
- Unifies intuitionistic (q=1) and Pythagorean (q=2)

### VIKOR Method
Compromise ranking based on:
- S: Group utility (weighted sum)
- R: Individual regret (max deviation)
- Q: Compromise score (combination of S and R)

## 📄 License

MIT License

## 👨‍💻 Author

Generated as part of advanced AI automation system development for supply chain resilience research.
