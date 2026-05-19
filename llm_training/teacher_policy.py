### This will generate training and validation data set using teacher policy

import random
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import os

REGIONS = ["single_cell", "cluster", "tissue_like"]

def generate_features() -> Dict[str, Any]:
    return {
        "region": random.choice(REGIONS),
        "population_pct": random.uniform(0, 100),
        "interaction_std": random.uniform(0, 1)
    }
def teacher_policy(features):
    pop = features["population_pct"]
    std = features["interaction_std"]

    reasoning = []
    mode = "normal"

    # ACTION
    action = "stay"

    # LOW POPULATION
    if pop < 10:
        reasoning.append("Low population region detected.")

        if std > 0.6:
            reasoning.append("High variability → unstable region.")
        else:
            reasoning.append("Low variability → inactive region.")
            action = "move"

    # MEDIUM POPULATION
    elif pop < 50:
        reasoning.append("Medium population region detected.")

        if std < 0.1:
            reasoning.append("Weak interaction signal.")
            action = "move"
        elif std < 0.6:
            reasoning.append("Moderate activity detected.")
        else:
            reasoning.append("Highly dynamic region.")

    # HIGH POPULATION
    else:
        reasoning.append("High population region detected.")

        if std > 0.7:
            mode = "post_analysis"
            reasoning.append("Highly population and unstable → trigger post analysis mode.")
        else:
            reasoning.append("Highly population → trigger post analysis mode.")

    return action, mode, reasoning
#############    generate training chain
def tool_observation(features):
    return {
        "region": features["region"],
        "population_pct": round(features["population_pct"], 2),
        "interaction_std": round(features["interaction_std"], 3)
    }
def build_trajectory():
    features = generate_features()
    obs = tool_observation(features)

    action1, mode1, reasoning1 = teacher_policy(features)

    user_msg = (
        f"Region: {obs['region']}\n"
        f"Population: {obs['population_pct']}%\n"
        f"Interaction STD: {obs['interaction_std']}"
    )

    assistant_1 = (
        "Thought: " + reasoning1[0] + "\n"
        "Action: analyze_features"
    )

    tool_msg = json.dumps(obs)

    assistant_2 = (
        "Thought: " + " ".join(reasoning1) + "\n"
        f"Mode: {mode1}\n"
        f"Action: {action1}"
    )

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_1},
            {"role": "tool", "content": tool_msg},
            {"role": "assistant", "content": assistant_2}
        ]
    }

def generate_dataset(n=100):
    return [build_trajectory() for _ in range(n)]
def save_jsonl(dataset: List[Dict[str, Any]], path: str = "dataset.jsonl"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 1. Set a random seed so your splits are reproducible
    random.seed(42) 
    
    # 2. Generate a larger pool of data (e.g., 120 samples)
    total_samples = 2000
    data = generate_dataset(total_samples)
    
    # 3. Calculate split index (90% training, 10% validation)
    split_idx = int(total_samples * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 4. Save into separate files
    os.makedirs("LLM_fine_tuning", exist_ok=True)
    save_jsonl(train_data, path="LLM_fine_tuning/train_dataset.jsonl")
    save_jsonl(val_data, path="LLM_fine_tuning/val_dataset.jsonl")
    
    print(f"Total Generated: {total_samples}")
    print(f"Saved Training Samples: {len(train_data)} to LLM_fine_tuning/train_dataset.jsonl")
    print(f"Saved Validation Samples: {len(val_data)} to LLM_fine_tuning/val_dataset.jsonl")
