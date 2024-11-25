# Schema Guided Dialogue Dataset - Natural Language Generation

This is the repository for NLG research on Schema Guided Dialogue Conversational AI.

# Technology 

- framework: pytorch lightning
- pretrained/datasets/metrics: HuggingFace

# Organization

## General PytorchModule, Training and Testing

The `models/` directory contains config file for training and testing the model, it is also where checkpoints and logs are saved.

The `train_model.py`, `infer_model.py`, `mark.py` are python scripts for training, making inference, and marking results. They SHOULD NOT be called directly but act as dependencies of `ectrain`, `ecinfer` and `ecmark` . Their behaviour is configurable  via the `config.yaml` file other each model folder.

The `sbatch_infer.sh`, `sbatch_train.sh` files are for submitting jobs to slurm  clusters. They should not be called as well.

`transformer_modules.py` defines the `LightningModule` which contains the behavior of the model. Anything relating to the training, validation, logging and inference of the model should be defined here as a single class.

`data_modules.py` defines the `LightningDataModule` which defines how data is preprocessed, stored and loaded. Everything should be implemented as a single class to be referenced by the `config.yaml` file.

## Prefix-Tuning

We implement the models for prefix-tuning GPT2/T5 in `prefix_models.py`. They represent the combination of prefix model (MLP) + Language Model (GPT2/T5). These models are prefixed by `PT-`. The neural networks that are responsible for producing the prefixes are defined in `modeling_prefixes.py` (e.g. `MLPPrefixModel`). Since PT cannot be directly implemented with the T5 model in the `transformers` library, we extend the T5 transformer classes in `modeling_prefix_t5.py`, following the naming convention of the `transformers` library. Note that there is no `modeling_prefix_gpt2.py` because PT can be implemented on GPT2 by passing the prefixes as `past_key_values`.

The PytorchLightningModules for the PT models are implemented in `transformer_modules.py`.

# Workflow

1. To **develop a new model**, write a new class in `transformer_modules.py` to encapsulate its behaviour. 

2. To **use a new dataset**, write a new class in `data_modules.py` to define how it is read/processed/loaded

3. To train/infer/eval a new model:, first create a folder under `models/`. e.g `models/t5-base-mymodel`

4. **Configuration**: Write the `config.yaml` file under `models/<model-name>` to define the training and testing configurations. You may reference the templates in existing models. 

> Make sure `<model_name>` is unique!

5. **Training**: Double check your `models/<model_name>/config.yaml` file. Run `ectrain <model_name>` to train the model on slurm, or `ectrain <model_name> local` to train it locally to train it locally. You will find checkpoints and logs (default tensorboard) under `models/<model_name>/logs/<exp_name>/<version>/`

6. **Inference**: Double check your `models/<model_name>/config.yaml` file. Run `ecinfer <model_name>` to run inference. The result will be found in the `decode_path` of your choice

7. **Marking**: Run `ecmark <model_name> <exp_name> <version>` to get a profile of metrics on the decoded results. The program will look for `test_cases.txt` in `models/<model_name>/logs/<exp_name>/<version>/`. It will print a table and save a csv file with the results

# Configuration File
The configuration file is the `config.yaml` under each model's root directory. It controls the behavior of *training* and *testing*. The configuration will be copied to the experiment directory once `ectrain` is run.

For examples of configuration file, please consult the example models under `models/`. The following gives an idea of what a config file looks like.

```yaml
---
# Meta Info
  ModelName: "t5-small-SGD"
  Note: "T5-small for SGD, using naive linearization"

# LightningModule
  LightningModuleName: "HFT5GenerationModel"
  LightningModuleParas:
    model_path: "t5-small"

# LightningDataModule
  LightningDataModuleName: "GEMSGD_DataModule"
  LightningDataModuleParas:
    batch_size: 8
    tokenizer_path: "t5-small"
    tokenizer_class: T5Tokenizer
    force_process: false
    save_cache: true
    encode_args:
      padding: max_length
      truncation: true

# Training
  TrainerParas:
    accumulate_grad_batches: 1
    max_epochs: 3
    val_check_interval: 10000
  CheckpointStepInterval: 10000

# Logging
  TrainLoggerName: TensorBoardLogger
  TrainLoggerParas:
    name: naive-linear
    version: train-test

  TestLoggerName: CSVLogger
  TestLoggerParas:
    name: naive-linear
    version: test-1-23122

# Testing
  LoadingParas:
    checkpoint_path: "logs/naive-linear/train-2/checkpoints/epoch=1-step=23122.ckpt"
    save_decode: True
    decode_path: logs/naive-linear/test-1-23122
```

## MetaInfo
Meta information

## LightningModule
Configure which `LightningModule` in `transformer_modules.py` will be used for `train_model.py` and `test_model.py`, as well as its initialization parameters.

## LightningDataModule
Configure which `LightningDataModule` in `data_modules.py` will be used for `train_model.py` and `test_model.py`, as well as its initialization parameters.

## Logging
Define what logging class will be used and the **experiment name and version** which **defines** the folder into which training and inference results will be saved

## Testing
Configure the arguments of `infer_model.py` and which model to test.

