### This will generate training and validation data set using teacher policy

import random
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
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
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    data = generate_dataset(100)
    save_jsonl(data, path="/content/drive/MyDrive/LLM_fine_tuning/dataset.jsonl")

    print("Generated samples:", len(data))
    print("Example:", data[0])
