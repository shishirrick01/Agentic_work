# GenAI-Driven Black Swan Scenario Generation + Fuzzy MCDM System

An advanced AI automation system that generates unprecedented supply chain disruption scenarios using Large Language Models and evaluates resilience strategies using Pythagorean Fuzzy MCDM methods.

## 🚀 Features

- **LLM-powered Scenario Generation**: Creates 1,000+ highly specific, multi-node supply chain disruption scenarios
- **NLP Translation Layer**: Converts qualitative scenarios into quantitative parameters
- **Pythagorean Fuzzy TOPSIS**: Advanced MCDM method for ranking resilience strategies
- **q-Rung Orthopair Fuzzy VIKOR**: Alternative MCDM method for robust decision-making
- **Synthetic Data Generation**: Solves the "cold start" problem in supply chain AI

## 📁 Project Structure

```
/workspace/
├── scenario_generator/     # LLM-based scenario generation
├── nlp_translator/        # Qualitative to quantitative translation
├── fuzzy_mcdm/           # Fuzzy MCDM methods (TOPSIS, VIKOR)
├── data/                 # Input data and generated scenarios
├── config/               # Configuration files
├── utils/                # Utility functions
├── main.py              # Main execution script
└── requirements.txt      # Dependencies
```

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Usage

```bash
# Generate scenarios and evaluate strategies
python main.py --num_scenarios 1000 --method pythagorean_topsis

# Only generate scenarios
python main.py --mode generate --num_scenarios 500

# Only evaluate strategies
python main.py --mode evaluate --method fuzzy_vikor
```

## 📊 Methodology

### 1. Scenario Generation
- Input: Macro-economic reports, climate data, geopolitical news
- Output: 1,000+ unprecedented disruption scenarios
- Example: "Cyberattack on Port of Rotterdam + Rare earth shortage in Southeast Asia"

### 2. NLP Translation
- Extracts quantitative parameters from text scenarios
- Maps to: Lead time changes, cost increases, capacity reductions

### 3. Fuzzy MCDM Evaluation
- Evaluates resilience strategies: multi-sourcing, safety stock, nearshoring
- Uses Pythagorean Fuzzy Sets for uncertainty handling
- Ranks strategies based on performance across all scenarios

## 🎓 Academic Value

This system addresses the critical gap in traditional scenario planning by:
- Generating **unprecedented** events (no historical bias)
- Using **GenAI for synthetic data** in Operations Research
- Applying **advanced fuzzy mathematics** for real-world uncertainty
- Solving the **"cold start" problem** in supply chain AI

## 📄 License

MIT License