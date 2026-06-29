"""
Prompt Templates for Black Swan Scenario Generation

These templates guide the LLM to generate unprecedented, multi-node
supply chain disruption scenarios with specific details.
"""

from typing import Dict, List


class ScenarioPromptTemplates:
    """Collection of prompt templates for scenario generation."""
    
    BASE_SYSTEM_PROMPT = """You are an expert supply chain risk analyst and futurist. 
Your task is to generate unprecedented "Black Swan" disruption scenarios that have 
never occurred before but are plausible given current global trends.

Focus on:
1. Multi-node disruptions (affecting multiple points in supply chains)
2. Compound events (2+ simultaneous or cascading disruptions)
3. Specific geographic locations and industries
4. Quantifiable impacts where possible

Avoid generic scenarios like "natural disaster" or "economic crisis". 
Be highly specific and creative."""

    SCENARIO_GENERATION_TEMPLATE = """
Context Information:
{context_data}

Generate {num_scenarios} unique Black Swan supply chain disruption scenarios.

Each scenario should include:
- TITLE: A concise, descriptive title
- DESCRIPTION: Detailed narrative (3-5 sentences)
- AFFECTED_NODES: List of specific supply chain nodes (ports, factories, regions)
- DISRUPTION_TYPES: Types of disruptions (cyber, climate, geopolitical, etc.)
- ESTIMATED_DURATION: How long the disruption lasts
- SEVERITY: Low/Medium/High/Extreme

Format each scenario as JSON:
{{
    "id": "scenario_001",
    "title": "...",
    "description": "...",
    "affected_nodes": ["node1", "node2"],
    "disruption_types": ["type1", "type2"],
    "estimated_duration_days": 30,
    "severity": "High",
    "root_causes": ["cause1", "cause2"]
}}

Return ONLY a JSON array of scenarios, no additional text.
"""

    CLIMATE_FOCUSED_TEMPLATE = """
Generate climate-related Black Swan scenarios considering:
- Extreme weather events beyond historical precedents
- Tipping points in climate systems
- Cascading effects from ecosystem collapse
- Infrastructure failures due to climate stress

Context: {climate_data}
"""

    GEOPOLITICAL_FOCUSED_TEMPLATE = """
Generate geopolitically-driven Black Swan scenarios considering:
- Unexpected alliance shifts or conflicts
- Trade war escalations
- Sanctions on critical resources
- Political instability in key regions

Context: {geopolitical_data}
"""

    TECHNOLOGY_FOCUSED_TEMPLATE = """
Generate technology-related Black Swan scenarios considering:
- Cyberattacks on critical infrastructure
- AI system failures
- Semiconductor supply collapses
- Communication network blackouts

Context: {technology_data}
"""

    COMPOUND_EVENT_TEMPLATE = """
Generate compound Black Swan scenarios where MULTIPLE disruptions occur simultaneously:

Primary Event: {primary_event_type}
Secondary Event: {secondary_event_type}
Tertiary Event (optional): {tertiary_event_type}

Show how these events cascade and amplify each other's impacts.
"""

    @classmethod
    def get_generation_prompt(
        cls,
        context_data: str,
        num_scenarios: int = 10,
        focus_area: str = "general"
    ) -> Dict[str, str]:
        """
        Generate a complete prompt for scenario generation.
        
        Args:
            context_data: Background information (economic, climate, geopolitical data)
            num_scenarios: Number of scenarios to generate
            focus_area: One of 'general', 'climate', 'geopolitical', 'technology'
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        system_prompt = cls.BASE_SYSTEM_PROMPT
        
        if focus_area == "climate":
            user_template = cls.CLIMATE_FOCUSED_TEMPLATE
            user_prompt = user_template.format(climate_data=context_data)
        elif focus_area == "geopolitical":
            user_template = cls.GEOPOLITICAL_FOCUSED_TEMPLATE
            user_prompt = user_template.format(geopolitical_data=context_data)
        elif focus_area == "technology":
            user_template = cls.TECHNOLOGY_FOCUSED_TEMPLATE
            user_prompt = user_template.format(technology_data=context_data)
        else:
            user_template = cls.SCENARIO_GENERATION_TEMPLATE
            user_prompt = user_template.format(
                context_data=context_data,
                num_scenarios=num_scenarios
            )
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }

    @classmethod
    def get_refinement_prompt(cls, scenarios: List[Dict]) -> Dict[str, str]:
        """
        Generate a prompt to refine and diversify existing scenarios.
        
        Args:
            scenarios: List of previously generated scenarios
            
        Returns:
            Dictionary with refinement prompts
        """
        system_prompt = """You are refining Black Swan scenarios to ensure maximum diversity 
and unprecedented nature. Remove clichés, increase specificity, and ensure no two scenarios 
are similar."""
        
        user_prompt = f"""Review these {len(scenarios)} scenarios and identify:
1. Scenarios that are too generic or historically common
2. Missing combinations of disruption types
3. Underrepresented geographic regions
4. Opportunities for more extreme but plausible events

Existing scenarios:
{scenarios}

Suggest 10 improved or entirely new scenarios that address these gaps."""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
