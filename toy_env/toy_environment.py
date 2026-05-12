#### toy environment to run fine-tuned llm

import time
import random
import json
import matplotlib.pyplot as plt

### we are using a toy environment to show the performance of  fine-tuned LLM
class ToyMicroscopyEnv:
    def __init__(self, delay=0.5):
        self.delay = delay

    def generate_observation(self):
        return {
            "region": random.choice(["single_cell", "cluster", "tissue_like"]),
            "population_pct": random.uniform(0, 100),
            "interaction_std": random.uniform(0, 1)
        }

    def call_tool(self, action):
        # simulate compute / microscope delay
        time.sleep(self.delay)

        obs = self.generate_observation()

        return {
            "action_received": action,
            "observation": obs,
            "status": "tool_completed"
        }
def run_episode(model, tokenizer, env, user_prompt):
    history = []

    prompt = user_prompt

    for step in range(3):  # multi-step rollout
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # simulate tool call
        action = "stay"  # in real version parse from response

        tool_result = env.call_tool(action)

        # build next prompt
        prompt = response + "\nToolResult: " + str(tool_result)

        history.append({
            "step": step,
            "response": response,
            "tool_result": tool_result
        })

    return history


def save_trace(trace, path="run_trace.json"):
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)


def visualize_trace(trace):
    steps = []
    pop = []

    for t in trace:
        obs = t["tool_result"]["observation"]
        steps.append(t["step"])
        pop.append(obs["population_pct"])

    plt.plot(steps, pop)
    plt.title("Agent perception over time")
    plt.xlabel("Step")
    plt.ylabel("Population %")
    plt.show()
