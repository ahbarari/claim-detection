import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger

logger.add("evaluation.log", rotation="500 MB")

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


class AdapterTransformerEvaluator:
    def __init__(self, pretrained_model, task_adapter_paths):
        self.pretrained_model = pretrained_model
        self.task_adapter_paths = task_adapter_paths

    def evaluate(self, test_data, model_path, attention_scores_output, random_seed):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, max_length=128, truncation=True, padding=True)
        model = AutoAdapterModel.from_pretrained(self.pretrained_model)

        for task_adapter_path in self.task_adapter_paths:
            model.load_adapter(task_adapter_path, set_active=True)

        model.load_adapter_fusion(str(model_path), set_active=True)

        all_attention_scores = []

        for text in list(test_data["text"].unique()):
            try:
                # Tokenize the input
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

                # Move inputs to GPU if available
                inputs = {key: val.to(model.device) for key, val in inputs.items()}

                # Perform forward pass and retrieve attention scores
                outputs = model(**inputs, output_adapter_fusion_attentions=True)
                attention_scores = outputs.adapter_fusion_attentions

                # Store attention scores
                all_attention_scores.append(attention_scores)

            except IndexError as e:
                logger.error(f"Error processing text: {text}. Error: {e}")

        # Save attention scores for further analysis
        torch.save(all_attention_scores, f"{attention_scores_output}/attention_scores.pt")

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--test_dataset_dir", type=str, required=True, help="Directory of the test dataset.")
    parser.add_argument("--test_claim", type=str, required=True, help="Name of the test claim.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the adapter fusion model.")
    parser.add_argument("--task_adapter_paths", type=lambda s: [i for i in s.split(',')],
                        required=True, help="Comma-separated list of task adapter paths.")
    parser.add_argument("--attention_scores_output", type=str, required=True,
                        help="Path to save the attention scores file.")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for reproducibility.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)

    test_dataset_dir = Path(args.test_dataset_dir)
    test_data = pd.read_csv(test_dataset_dir / f'{args.test_claim}_test.tsv', sep='\t', quoting=3)

    evaluator = AdapterTransformerEvaluator(
        pretrained_model=args.pretrained_model,
        task_adapter_paths=args.task_adapter_paths
    )

    logger.info(f"Evaluating with model at {args.model_path}")
    logger.info(f"Saving attention scores to {args.attention_scores_output}")

    evaluator.evaluate(
        test_data=test_data,
        model_path=Path(args.model_path),
        attention_scores_output=args.attention_scores_output,
        random_seed=args.random_seed
    )
