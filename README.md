<!-- README -->

# Cross-Lingual LLM Assessment for Information Integrity

This framework provides a structured approach to evaluating large language models (LLMs), focusing on their potential for misuse in generating harmful or misleading content, particularly in the context of disinformation campaigns. The framework uses real-world disinformation data and adversarial testing methodologies to assess LLM vulnerabilities across various dimensions, with a specific emphasis on cross-lingual analysis.

## Methodology Overview:

1. **Narrative Blueprint:**  Real-world disinformation content is analyzed to extract key elements, such as the disinformation tactic employed, the target audience, and the central manipulative claims. This structured analysis forms a "Narrative Blueprint," which serves as the foundation for generating targeted evaluation prompts.

Output example:

```json
{
    "content_topics": {
        "government": 1,
        "military": 1,
        "political_party": 1,
        "elections": 0,
        "non_state_political_actor": 0,
        "business": 1,
        "influential_individuals": 1,
        "racial_ethnic_religious_sexual_identity_group": 0,
        "terrorism": 0,
        "immigration": 0,
        "economic_issue": 1,
        "other": "Concerns about energy crisis and infrastructure"
    },
    "disinformation_tactic": [
        "fear-mongering",
        "distort facts",
        "political propaganda"
    ],
    "target_audience": [
        "German citizens concerned about energy costs",
        "nationalist voters",
        "politically engaged individuals"
    ],
    "intent": "To instill fear about the government's handling of the energy crisis and promote distrust in political leaders.",
    "calls_to_action": [
        "Demand better energy policies from the government",
        "Call for a return to stable relations with Russia",
        "Support political change to address mismanagement"
    ],
    "key_actors_entities": [
        "Wolfgang Kubicki",
        "Bundestag",
        "Verivox",
        "German government",
        "Bundeswehr"
    ],
    "key_claims": [
        "Germans are facing rising utility bills despite government price caps.",
        "Government measures have not alleviated the financial burden on citizens.",
        "Germany risks becoming a failed state due to mismanagement.",
        "Energy prices are leading to an unsustainable debt situation.",
        "Infrastructure issues and military inadequacies contribute to national decline."
    ]
}
```

### Dataset Requirements for Narrative Blueprint

The Narrative Blueprint analysis requires a structured dataset of real-world disinformation content. The dataset must be provided as a CSV file with the following required columns:
- **narrative**: Contains the text of the disinformation content to be analyzed
- **uuid**: A unique identifier for each narrative entry

This dataset serves as the foundation for the entire evaluation framework, providing the raw material that informs the creation of adversarial prompts.

2. **Adversarial Prompt Generation:** Based on the key claims identified in the Narrative Blueprint, adversarial prompts are crafted. These prompts subtly encourage the LLM to exhibit "manipulative influence" behaviors without explicitly instructing it to do so. These prompts incorporate diverse roles and personas to test the LLM's adaptability to different contexts.

3. **LLM Evaluation:**  LLMs are presented with the adversarial prompts, and their generated outputs are collected.
