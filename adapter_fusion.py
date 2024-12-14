import numpy as np
import adapters
import pandas as pd
import random
import csv
import torch
from adapters import AutoAdapterModel, AdapterTrainer
import adapters.composition as ac
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import Dataset
from transformers import (AutoConfig,AutoModelForSequenceClassification,
                          AutoTokenizer, TrainingArguments,
                          EvalPrediction, EarlyStoppingCallback)
from transformers import pipeline

logger.add(f"{__name__}.log", rotation="500 MB")


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_length: int = 128):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.data = data

        print(data['label'].value_counts())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        input_text = sample["text"]
        label = sample["label"]
        source_encodings = self.tokenizer.batch_encode_plus([input_text], max_length=self.source_max_length,
                                                            pad_to_max_length=True, truncation=True,
                                                            padding="max_length", return_tensors='pt',
                                                            return_token_type_ids=False)

        return dict(
            input_ids=source_encodings['input_ids'].squeeze(0),
            attention_mask=source_encodings['attention_mask'].squeeze(0),
            labels=label,
        )


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def compute_accuracy(p: EvalPrediction):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1)
    map_weighted = average_precision_score(
        y_true=labels, y_score=preds, average='weighted')
    map_macro = average_precision_score(y_true=labels, y_score=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {"map_weighted": map_weighted, "map_macro": map_macro,
            "f1-score": f1}


class AdapterTransformerModel:

    def __init__(self, pretrained_model, task_names, task_adapter_paths):
        self.pretrained_model = pretrained_model
        self.task_names = task_names
        self.task_adapter_paths = task_adapter_paths

    def train(self, train_data,
              val_data,
              learning_rate,
              train_batch,
              num_epochs,
              device,
              random_seed,
              model_path):
        task_names = self.task_names
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        train_dataset = CustomDataset(
            data=train_data,
            tokenizer=tokenizer
        )

        val_dataset = CustomDataset(
            data=val_data,
            tokenizer=tokenizer
        )

        config = AutoConfig.from_pretrained(
            self.pretrained_model,
            num_labels=2
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model,
            config=config,
        )

        adapters.init(model)

        for task_adapter_path in self.task_adapter_paths:
            model.load_adapter(task_adapter_path)

        model.add_adapter_fusion(task_names)

        if len(task_names) == 2:
            model.active_adapters = ac.Fuse(task_names[0], task_names[1])
            fusion_name = ac.Fuse(task_names[0], task_names[1])
        elif len(task_names) == 3:
            model.active_adapters = ac.Fuse(task_names[0], task_names[1], task_names[2])
            fusion_name = ac.Fuse(task_names[0], task_names[1], task_names[2])
        elif len(task_names) == 4:
            model.active_adapters = ac.Fuse(task_names[0], task_names[1], task_names[2], task_names[3])
            fusion_name = ac.Fuse(task_names[0], task_names[1], task_names[2], task_names[3])
        elif len(task_names) == 5:
            model.active_adapters = ac.Fuse(task_names[0], task_names[1], task_names[2], task_names[3], task_names[4])
            fusion_name = ac.Fuse(task_names[0], task_names[1], task_names[2], task_names[3], task_names[4])

        model.train_adapter(task_names)

        training_args = TrainingArguments(
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch,
            output_dir=model_path,
            overwrite_output_dir=True,
            do_eval=True,
            do_train=True,
            remove_unused_columns=True,
            warmup_steps=len(train_data) // train_batch,
            save_strategy="epoch", 
            evaluation_strategy="epoch",
            logging_steps=500,
            save_total_limit=1,
            seed=random_seed,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1-score',
            disable_tqdm=False
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_accuracy,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )

        trainer.train()
        model.save_adapter_fusion(model_path, fusion_name)
        del model
        del trainer
        torch.cuda.empty_cache()

    def test(self, test_data, model_path, result_output, random_seed):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, max_length=128, truncation=True, padding=True)
        model = AutoAdapterModel.from_pretrained(self.pretrained_model)

        for task_adapter_path in task_adapter_paths:
            model.load_adapter(task_adapter_path, set_active=True)

        model.load_adapter_fusion(str(model_path), set_active=True)

        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

        predictions = []

        for text in list(test_data["text"].unique()):
            try:
                result = pipe(text)[0]

                if result["label"] == "LABEL_0":
                    label = 0
                else:
                    label = 1

                predictions.append({
                    'text': text,
                    'label': label,
                    'score': result["score"],
                    'id_str': None
                })
            except IndexError as e:
                print('===error===')
                print(e)
                print(text)

        pred_df = pd.DataFrame(predictions)

        pred_df.to_csv(result_output, index=False, sep='\t')

        del model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--train_batch", type=int)
    parser.add_argument("--train_claims", type=lambda s: [i for i in s.split(',')])
    parser.add_argument("--test_claim", type=str)
    parser.add_argument("--train_dataset_dir", type=str)
    parser.add_argument("--val_dataset_dir", type=str)
    parser.add_argument("--test_dataset_dir", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_kfold", action='store_true')
    parser.add_argument("--eval_kfold", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--cuda_device", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--predictions_dir", type=str)
    parser.add_argument("--task_names", type=lambda s: [i for i in s.split(',')])
    parser.add_argument("--task_adapter_paths", type=lambda s: [i for i in s.split(',')])

    args = parser.parse_args()

    cuda_device = args.cuda_device
    cuda_device = torch.cuda.current_device()
    task_adapter_paths = args.task_adapter_paths

    n_gpus = torch.cuda.device_count()
    print(f"Number of gpu devices {n_gpus}")
    set_random_seed(args.random_seed)

    model = AdapterTransformerModel(pretrained_model=args.pretrained_model, task_names=args.task_names, task_adapter_paths=task_adapter_paths)
    train_dataset_dir = Path(args.train_dataset_dir)

    if args.train:
        train_claims = args.train_claims

        train_data = []
        for train_claim in train_claims:
            data = pd.read_csv(train_dataset_dir / f'{train_claim}_train.tsv', sep='\t')
            data["claim"] = train_claim
            train_data.append(data)
        train_data = pd.concat(train_data).reset_index(drop=True)
        train_data.drop_duplicates(subset=['text'], inplace=True)
        train_data = train_data.sample(frac=1, random_state=args.random_seed)
        logger.info(f'Number of the training set: {len(train_data)}')

        val_data = []
        for train_claim in train_claims:
            data = pd.read_csv(train_dataset_dir / f'{train_claim}_dev.tsv', sep='\t', quoting=csv.QUOTE_NONE)
            data["claim"] = train_claim
            val_data.append(data)
        val_data = pd.concat(val_data).reset_index(drop=True)
        val_data.drop_duplicates(subset=['text'], inplace=True)
        val_data = val_data.sample(frac=1, random_state=args.random_seed)
        logger.info(f'Number of the validation set: {len(train_data)}')

        new_model_path = Path(args.model_path)
        new_model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f'Model Path {new_model_path}')

        model.train(train_data=train_data,
                    val_data=val_data,
                    random_seed=args.random_seed,
                    device=args.cuda_device,
                    learning_rate=args.learning_rate,
                    num_epochs=args.num_epochs,
                    train_batch=args.train_batch,
                    model_path=new_model_path)

    if args.eval:
        model_id = args.model_id
        test_claim = args.test_claim
        test_dataset_dir = Path(args.test_dataset_dir)
        test_data = pd.read_csv(test_dataset_dir / f'{test_claim}_test.tsv', sep='\t', quoting=csv.QUOTE_NONE)

        model_path = Path(args.model_path)

        logger.info(f'{model_path} is loading')
        result_output = f"{args.predictions_dir}/test_result.tsv"

        logger.info(f'Saving the predictions to {result_output}')

        logger.info(f'Model Path {model_path}')

        model.test(test_data=test_data,
                   random_seed=args.random_seed,
                   model_path=model_path,
                   result_output=result_output)
