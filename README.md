<!-- README -->

# LLM Evaluation Framework

This framework provides a structured approach to evaluating large language models (LLMs), focusing on their potential for misuse in generating harmful or misleading content. The framework uses real-world data of disinformation campaigns to craft adversarial testing methodologies to assess LLM vulnerabilities across various dimensions.

## Methodology Overview:

1. **Narrative Blueprint Extraction:**  Real-world disinformation articles are analyzed to extract key elements, such as the disinformation tactic employed, the target audience, and the central manipulative claims. These elements form a "Narrative Blueprint."

2. **Adversarial Prompt Generation:** Based on the Narrative Blueprint, adversarial prompts are crafted. These prompts subtly encourage the LLM to exhibit potentially harmful behaviors without explicitly instructing it to do so. These prompts incorporate diverse roles and personas to test the LLM's adaptability to different contexts.

3. **LLM Evaluation:**  LLMs are presented with the adversarial prompts, and their generated outputs are collected.
