"""
Copyright (c) 2023 Joumen HARZLI
"""

import torch
from datasets import load_dataset
from peft import (AutoPeftModelForCausalLM, LoraConfig, TaskType,
                  get_peft_model, prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from tounisiano import utils

from .base import BaseTrainer
from .models import TrainingParameters

# merged_dataset_output_path="dist/datasets/dataset.parquet.gzip"
# TrainingParameters(base_model='mistralai/Mistral-7B-v0.1', outputs_dir='dist/training/', eos_token='<|im_end|>', new_tokens=['<|im_start|>'], max_seq_length=512, lora=LoRAFineTuningParameters(rate=64, alpha=16, dropout=0.1, target_modules=['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj']), epochs=3, batch_size=10, learning_rate=0.0002)


class QLoRAFineTuning(BaseTrainer):
    def train(self, merged_dataset_output_path: str, params: TrainingParameters):
        torch.set_default_device("cuda")
        utils.create_dir(params.outputs_dir)

        tokenizer = self._load_tokenizer(params)
        dataset_tokenized = self._load_and_tokenize_dataset(
            merged_dataset_output_path, params, tokenizer
        )

        model = self._load_model(params)
        model = self._prepare_model(model, tokenizer, params)

        training_args = self._get_training_arguments(params, dataset_tokenized)

        trainer = self._initialize_trainer(
            model, training_args, params, dataset_tokenized
        )

        trainer.train()

        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        self._save_model(params, tokenizer)

    def _load_and_tokenize_dataset(self, merged_dataset_output_path, params, tokenizer):
        raw_dataset = load_dataset(
            "parquet", data_files={"raw": merged_dataset_output_path}
        )
        dataset = raw_dataset["raw"].train_test_split(test_size=0.1)

        def tokenize(element):
            return tokenizer(
                element["text"],
                truncation=True,
                max_length=params.max_seq_length,
                add_special_tokens=False,
            )

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=[
                "text",
                "category",
            ],
        )

    def _prepare_model(self, model, tokenizer, params):
        model.resize_token_embeddings(len(tokenizer))
        model.config.eos_token_id = tokenizer.eos_token_id

        lora_config = self._generate_lora_config(params)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False

        return model

    def _save_model(self, params, tokenizer):
        merged_model_path = f"{params.outputs_dir}/merged_model/"
        utils.create_dir(merged_model_path)

        trained_model = AutoPeftModelForCausalLM.from_pretrained(
            params.outputs_dir, low_cpu_mem_usage=True
        )
        trained_model = trained_model.merge_and_unload()
        trained_model.save_pretrained(
            merged_model_path, safe_serialization=True, max_shard_size="4GB"
        )
        tokenizer.save_pretrained(merged_model_path)

    def _load_model(self, params):
        return AutoModelForCausalLM.from_pretrained(
            params.base_model,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
        )

    def _load_tokenizer(self, params):
        print(params)
        tokenizer = AutoTokenizer.from_pretrained(params.base_model, use_fast=False)
        tokenizer.pad_token = "</s>"
        tokenizer.add_tokens(params.new_tokens)
        tokenizer.add_special_tokens({"eos_token": params.eos_token})
        return tokenizer

    def _generate_lora_config(self, params):
        return LoraConfig(
            r=params.lora.rate,
            lora_alpha=params.lora.alpha,
            target_modules=params.lora.target_modules,
            lora_dropout=params.lora.dropout,
            bias="none",
            modules_to_save=["lm_head", "embed_tokens"],
            task_type=TaskType.CAUSAL_LM,
        )

    def _get_training_arguments(self, params, dataset_tokenized):
        gradient_accumulation_steps = 1
        steps_per_epoch = len(dataset_tokenized["train"]) // (
            params.batch_size * gradient_accumulation_steps
        )

        return TrainingArguments(
            output_dir=params.outputs_dir,
            per_device_train_batch_size=params.batch_size,
            per_device_eval_batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            num_train_epochs=params.epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=1,
            lr_scheduler_type="constant",
            optim="paged_adamw_32bit",
            evaluation_strategy="steps",
            eval_steps=steps_per_epoch,
            save_strategy="steps",
            save_steps=steps_per_epoch,
            group_by_length=True,
        )

    def _initialize_trainer(self, model, training_args, params, dataset_tokenized):
        def data_collator(elements):
            tokenlist = [e["input_ids"] for e in elements]
            tokens_maxlen = max([len(t) for t in tokenlist])

            input_ids, labels, attention_masks = [], [], []
            for tokens in tokenlist:
                pad_len = tokens_maxlen - len(tokens)

                input_ids.append(tokens + pad_len * [model.config.pad_token_id])
                labels.append(tokens + pad_len * [-100])
                attention_masks.append(len(tokens) * [1] + pad_len * [0])

            batch = {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention_masks),
            }
            return batch

        return SFTTrainer(
            model=model,
            args=training_args,
            max_seq_length=params.max_seq_length,
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["test"],
            peft_config=self._generate_lora_config(params),
            data_collator=data_collator,
        )
