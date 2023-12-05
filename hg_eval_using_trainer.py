from transformers import TrainingArguments, ViTForImageClassification, Trainer, DefaultDataCollator, ViTImageProcessor
from datasets import load_dataset

pretrained_model = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(pretrained_model)
image_processor = ViTImageProcessor.from_pretrained(pretrained_model)

training_args = TrainingArguments(
    # output_dir="google/vit-base-patch16-224-in21k",
    output_dir="",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

data_collator = DefaultDataCollator()
# https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/loading_methods#datasets.load_dataset
iterable_dataset = load_dataset('imagefolder', data_dir="../msn_shoeprint_retrieval/imagenet", streaming=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {"accuracy": (predictions == labels).float().mean().item()}


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=iterable_dataset["train"],
    eval_dataset=iterable_dataset["validation"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.evaluate()

# https://huggingface.co/docs/evaluate/a_quick_tour

# if output_dir is not defined, I get 404 Client error: 
# https://discuss.huggingface.co/t/repositorynotfounderror-404-client-error/52908