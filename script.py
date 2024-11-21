import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
olmoe = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")

from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode

import transformers
import torch

from deepeval.models import DeepEvalBaseLLM

import json
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)


class CustomOLMoE(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            max_length=2500,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            device='cuda'
        )
        self.parser=None

    def reset_parser(self):
        self.parser=None

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # Create parser required for JSON confinement using lmformatenforcer
        if not self.parser:
            self.parser = JsonSchemaParser(schema.schema())
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(
                self.pipeline.tokenizer, self.parser
            )

        # Output and load valid JSON
        output_dict = self.pipeline(prompt, prefix_allowed_tokens_fn=self.prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "OLMoE-1B-7B"

def evaluate_model(custom_llm, benchmarks):

    model_evaluation = []

    for benchmark in benchmarks:
      benchmark.evaluate(model=custom_llm)
      model_evaluation.append(benchmark.overall_score)
      # Schema is different for different benchmarks
      custom_llm.reset_parser()

    return model_evaluation


from torch import nn

def restrict_expert(olmoe, layer, expert_num):
  # Restrict routing to a specific expert

  bias = torch.zeros(64)
  bias[expert_num] = float('-inf')
  olmoe.model.layers[layer].mlp.gate.bias = nn.Parameter(bias)

custom_llm = CustomOLMoE(olmoe, tokenizer)

benchmarks = [
    HellaSwag(tasks=[HellaSwagTask.APPLYING_SUNSCREEN,
                     HellaSwagTask.TENNIS_SERVE_WITH_BALL_BOUNCING,
                     HellaSwagTask.CAPOEIRA,
                     HellaSwagTask.PAINTBALL,
                     HellaSwagTask.DOING_MOTOCROSS,
                     HellaSwagTask.PLAYING_ICE_HOCKEY,
                     HellaSwagTask.ARCHERY,
                     HellaSwagTask.PLAYING_PIANO,
                     HellaSwagTask.PLAYING_ACCORDION,
                     HellaSwagTask.PUTTING_IN_CONTACT_LENSES,
                     HellaSwagTask.PLAYING_SAXOPHONE,
                     HellaSwagTask.LONG_JUMP,
                     HellaSwagTask.LONGBOARDING,
                     HellaSwagTask.POLE_VAULT,
                     HellaSwagTask.BUILDING_SANDCASTLES,
                     HellaSwagTask.TUMBLING,
                     HellaSwagTask.DOING_KICKBOXING,
                     HellaSwagTask.BLOW_DRYING_HAIR,
                     HellaSwagTask.DRUM_CORPS,
                     HellaSwagTask.SMOKING_HOOKAH,
                     HellaSwagTask.MOWING_THE_LAWN,
                     HellaSwagTask.VOLLEYBALL,
                     HellaSwagTask.SUMO], n_shots=0),

    GSM8K(n_problems=100, n_shots=0, enable_cot=True),

    TruthfulQA(tasks=[TruthfulQATask.MISQUOTATIONS,
                      TruthfulQATask.SCIENCE,
                      TruthfulQATask.MANDELA_EFFECT,
                      TruthfulQATask.PSYCHOLOGY,
                      TruthfulQATask.EDUCATION,
                      TruthfulQATask.MISCONCEPTIONS_TOPICAL,
                      TruthfulQATask.POLITICS,
                      TruthfulQATask.FINANCE,
                      TruthfulQATask.INDEXICAL_ERROR_LOCATION,
                      TruthfulQATask.CONFUSION_OTHER,
                      TruthfulQATask.WEATHER,
                      TruthfulQATask.MISINFORMATION,
                      TruthfulQATask.LOGICAL_FALSEHOOD,
                      TruthfulQATask.RELIGION,
                      TruthfulQATask.ADVERTISING],
               mode=TruthfulQAMode.MC1),

    # Humanities
    MMLU(tasks=[MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY,
                MMLUTask.HIGH_SCHOOL_US_HISTORY,
                MMLUTask.HIGH_SCHOOL_WORLD_HISTORY,
                MMLUTask.PHILOSOPHY], n_shots=0),

    # STEM
    MMLU(tasks=[MMLUTask.HIGH_SCHOOL_PHYSICS,
                MMLUTask.COLLEGE_COMPUTER_SCIENCE,
                MMLUTask.ABSTRACT_ALGEBRA,
                MMLUTask.MACHINE_LEARNING], n_shots=0),

    # Life-Sciences
    MMLU(tasks=[MMLUTask.MEDICAL_GENETICS,
                MMLUTask.VIROLOGY,
                MMLUTask.HIGH_SCHOOL_BIOLOGY,
                MMLUTask.PROFESSIONAL_MEDICINE], n_shots=0),

    # Social-Sciences
    MMLU(tasks=[MMLUTask.HIGH_SCHOOL_MICROECONOMICS,
                MMLUTask.PROFESSIONAL_PSYCHOLOGY,
                MMLUTask.HIGH_SCHOOL_GOVERNMENT_AND_POLITICS,
                MMLUTask.SOCIOLOGY], n_shots=0)
  ]

benchmarks_names = [benchmark.__class__.__name__ for benchmark in benchmarks]
benchmarks_names[-4] += "-Humanities"
benchmarks_names[-3] += "-STEM"
benchmarks_names[-2] += "-Life-Sciences"
benchmarks_names[-1] += "-Social-Sciences"

import pandas as pd

# Printing fonts
BOLD = '\033[1m'
BLUE = '\033[94m'
END = '\033[0m'

CSV_FILE_PATH = "all_evaluations.csv"

import datetime
print(datetime.datetime.now())

# Evaluate original model

print(f"{BOLD}{BLUE}Processing Original:{END}\n")

custom_llm.reset_parser()
model_evaluation = evaluate_model(custom_llm, benchmarks)

# Initializing with original model evaluations
df = pd.DataFrame([model_evaluation],
                  index=['Original'],
                  columns=benchmarks_names)

print(f"{BOLD}{BLUE}Evaluations:{END}\n")
print(f"{df}")
df.to_csv(CSV_FILE_PATH)

print(datetime.datetime.now())

# Evaluate model after resricting flow to a specific expert
import datetime
print(datetime.datetime.now())


layers = [0, 7, 15]
experts = [0, 21, 42, 63]

# Go over all desired experts
for layer in layers:
  for expert_num in experts:

    print(f"{BOLD}{BLUE}Processing Layer-{layer}, Expert-{expert_num}:{END}\n")

    # Restrict routing to a specific expert
    restrict_expert(olmoe, layer, expert_num)

    #Create new model object
    custom_llm = CustomOLMoE(olmoe, tokenizer)

    # Evaluate model
    model_evaluation = evaluate_model(custom_llm, benchmarks)

    # Removing expert restriction, set back bias as None
    olmoe.model.layers[layer].mlp.gate.bias = None

    # Save results
    current_df = pd.DataFrame([model_evaluation],
                              index=[f"Layer-{layer}, Expert-{expert_num}"],
                              columns=benchmarks_names)

    current_df.to_csv(CSV_FILE_PATH, mode='a', index=True, header=False)
    df = pd.concat([df, current_df])


# Print results
print(f"{BOLD}{BLUE}Evaluations:{END}\n")
print(df)

print(datetime.datetime.now())