"""
Scenario Generator using LLMs

Generates Black Swan supply chain disruption scenarios by leveraging
Large Language Models with carefully crafted prompts.
"""

import json
import os
from typing import List, Dict, Optional, Union
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from .prompt_templates import ScenarioPromptTemplates


class ScenarioGenerator:
    """
    Generates unprecedented supply chain disruption scenarios using LLMs.
    
    Supports multiple LLM backends (OpenAI, local models via transformers)
    and can generate scenarios in batches with iterative refinement.
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 4000,
        output_dir: str = "data/scenarios"
    ):
        """
        Initialize the Scenario Generator.
        
        Args:
            llm_provider: LLM provider ('openai', 'huggingface', 'mock')
            model_name: Specific model to use
            api_key: API key for the LLM provider
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            output_dir: Directory to save generated scenarios
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client
        self.client = self._initialize_client(api_key)
        
        logger.info(f"ScenarioGenerator initialized with {llm_provider}/{model_name}")
    
    def _initialize_client(self, api_key: Optional[str]) -> Optional[object]:
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found. Using mock mode.")
                    return None
                return OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not installed. Using mock mode.")
                return None
        elif self.llm_provider == "huggingface":
            # Initialize Hugging Face pipeline
            try:
                from transformers import pipeline
                return pipeline("text-generation", model=self.model_name)
            except ImportError:
                logger.warning("transformers package not installed. Using mock mode.")
                return None
        else:
            logger.info("Using mock LLM for testing/demo")
            return None
    
    def generate_scenarios(
        self,
        num_scenarios: int = 100,
        context_data: Optional[Union[str, Dict]] = None,
        focus_areas: List[str] = None,
        batch_size: int = 10,
        refine: bool = True
    ) -> List[Dict]:
        """
        Generate Black Swan scenarios.
        
        Args:
            num_scenarios: Total number of scenarios to generate
            context_data: Background data (macro-economic, climate, geopolitical)
            focus_areas: List of focus areas ('general', 'climate', 'geopolitical', 'technology')
            batch_size: Number of scenarios per LLM call
            refine: Whether to iteratively refine scenarios
            
        Returns:
            List of generated scenario dictionaries
        """
        if focus_areas is None:
            focus_areas = ["general", "climate", "geopolitical", "technology"]
        
        if context_data is None:
            context_data = self._load_default_context()
        
        all_scenarios = []
        num_batches = (num_scenarios + batch_size - 1) // batch_size
        
        logger.info(f"Generating {num_scenarios} scenarios in {num_batches} batches")
        
        for batch_idx in tqdm(range(num_batches), desc="Generating scenarios"):
            remaining = num_scenarios - len(all_scenarios)
            current_batch_size = min(batch_size, remaining)
            
            # Rotate through focus areas for diversity
            focus_area = focus_areas[batch_idx % len(focus_areas)]
            
            # Prepare prompt
            prompts = ScenarioPromptTemplates.get_generation_prompt(
                context_data=context_data if isinstance(context_data, str) else json.dumps(context_data),
                num_scenarios=current_batch_size,
                focus_area=focus_area
            )
            
            # Generate scenarios
            batch_scenarios = self._call_llm(prompts)
            
            if batch_scenarios:
                all_scenarios.extend(batch_scenarios)
                logger.debug(f"Batch {batch_idx + 1}: Generated {len(batch_scenarios)} scenarios")
            
            # Rate limiting consideration
            if self.llm_provider == "openai" and (batch_idx + 1) % 10 == 0:
                import time
                time.sleep(1)  # Avoid rate limits
        
        # Refinement pass
        if refine and len(all_scenarios) > 0:
            logger.info("Refining scenarios for diversity and specificity...")
            all_scenarios = self._refine_scenarios(all_scenarios)
        
        # Save to file
        self._save_scenarios(all_scenarios)
        
        logger.success(f"Generated {len(all_scenarios)} total scenarios")
        return all_scenarios
    
    def _call_llm(self, prompts: Dict[str, str]) -> List[Dict]:
        """Call the LLM and parse the response."""
        if self.client is None:
            return self._generate_mock_scenarios(10)
        
        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompts["system"]},
                        {"role": "user", "content": prompts["user"]}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                response_text = response.choices[0].message.content.strip()
                
            elif self.llm_provider == "huggingface":
                result = self.client(
                    f"{prompts['system']}\n\n{prompts['user']}",
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                response_text = result[0]['generated_text']
            else:
                return self._generate_mock_scenarios(10)
            
            # Parse JSON response
            scenarios = self._parse_response(response_text)
            return scenarios
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_mock_scenarios(10)
    
    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response into scenario dictionaries."""
        try:
            # Try to extract JSON array
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            scenarios = json.loads(response_text)
            
            if isinstance(scenarios, dict):
                scenarios = [scenarios]
            elif isinstance(scenarios, list):
                pass
            else:
                logger.warning("Unexpected response format")
                return []
            
            # Validate and enrich scenarios
            validated = []
            for i, scenario in enumerate(scenarios):
                if isinstance(scenario, dict):
                    scenario = self._validate_scenario(scenario, i)
                    validated.append(scenario)
            
            return validated
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return []
    
    def _validate_scenario(self, scenario: Dict, index: int) -> Dict:
        """Validate and enrich a scenario dictionary."""
        # Ensure required fields
        required_fields = ['id', 'title', 'description', 'affected_nodes', 
                          'disruption_types', 'estimated_duration_days', 'severity']
        
        for field in required_fields:
            if field not in scenario:
                scenario[field] = self._get_default_value(field)
        
        # Add metadata
        scenario['generated_at'] = str(Path.cwd())
        scenario['validated'] = True
        
        return scenario
    
    def _get_default_value(self, field: str) -> any:
        """Get default value for missing fields."""
        defaults = {
            'id': f'scenario_{id(field)}',
            'title': 'Untitled Scenario',
            'description': 'No description provided',
            'affected_nodes': ['Unknown'],
            'disruption_types': ['Unknown'],
            'estimated_duration_days': 30,
            'severity': 'Medium'
        }
        return defaults.get(field, None)
    
    def _generate_mock_scenarios(self, count: int) -> List[Dict]:
        """Generate mock scenarios for testing/demo purposes."""
        import random
        
        locations = [
            "Port of Shanghai", "Port of Rotterdam", "Port of Los Angeles",
            "Suez Canal", "Panama Canal", "Strait of Hormuz",
            "Shenzhen Manufacturing Hub", "German Automotive Region",
            "Taiwan Semiconductor Facilities", "Australian Mining Region"
        ]
        
        disruption_types = [
            "Cyberattack", "Extreme Weather", "Geopolitical Sanction",
            "Labor Strike", "Infrastructure Failure", "Resource Shortage",
            "Terrorist Attack", "Pandemic Outbreak", "Financial Collapse",
            "Satellite Network Failure", "AI System Malfunction"
        ]
        
        severities = ["Low", "Medium", "High", "Extreme"]
        
        scenarios = []
        for i in range(count):
            num_nodes = random.randint(2, 4)
            selected_nodes = random.sample(locations, min(num_nodes, len(locations)))
            num_types = random.randint(2, 3)
            selected_types = random.sample(disruption_types, min(num_types, len(disruption_types)))
            
            scenario = {
                "id": f"mock_scenario_{i:04d}",
                "title": f"{selected_types[0]} at {selected_nodes[0]} with cascading effects",
                "description": f"A {selected_types[0].lower()} affects {selected_nodes[0]}, "
                              f"causing disruptions to {', '.join(selected_nodes[1:])}. "
                              f"This is compounded by {selected_types[1] if len(selected_types) > 1 else 'secondary effects'}.",
                "affected_nodes": selected_nodes,
                "disruption_types": selected_types,
                "estimated_duration_days": random.randint(7, 180),
                "severity": random.choice(severities),
                "root_causes": [f"Root cause {j}" for j in range(random.randint(1, 3))],
                "is_mock": True
            }
            scenarios.append(scenario)
        
        logger.debug(f"Generated {count} mock scenarios")
        return scenarios
    
    def _refine_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """Refine scenarios for diversity and quality."""
        if self.client is None:
            return scenarios
        
        prompts = ScenarioPromptTemplates.get_refinement_prompt(scenarios[:20])  # Limit for context
        
        try:
            refined = self._call_llm(prompts)
            if refined:
                # Combine original and refined, removing duplicates
                existing_titles = {s['title'] for s in scenarios}
                for scenario in refined:
                    if scenario.get('title') not in existing_titles:
                        scenarios.append(scenario)
                        existing_titles.add(scenario['title'])
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
        
        return scenarios
    
    def _load_default_context(self) -> Dict:
        """Load default context data if none provided."""
        return {
            "macroeconomic": "Global inflation trends, interest rate changes, trade imbalances",
            "climate": "Increasing frequency of extreme weather events, sea level rise, temperature anomalies",
            "geopolitical": "Rising tensions between major powers, resource nationalism, trade wars",
            "technological": "AI advancement, cybersecurity threats, semiconductor dependencies"
        }
    
    def _save_scenarios(self, scenarios: List[Dict], filename: Optional[str] = None) -> str:
        """Save scenarios to a JSON file."""
        if filename is None:
            timestamp = Path.cwd().strftime("%Y%m%d_%H%M%S") if hasattr(Path.cwd(), 'strftime') else "latest"
            filename = f"scenarios_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(scenarios)} scenarios to {filepath}")
        return str(filepath)
    
    def load_scenarios(self, filepath: str) -> List[Dict]:
        """Load scenarios from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
        logger.info(f"Loaded {len(scenarios)} scenarios from {filepath}")
        return scenarios
