# Mitigating Snowball Hallucinations in Small Language Models

This repository is the official implementation of research project Mitigating Snowball Hallucinations in Small Language Models of COMP 767, Fall 2024, McGill University.

## Requirements

To install requirements:

```setup
pip3 install torch torchvision torchaudio transformers json
```

## Inference/Testing

To test the model(s) in the paper, select the model script, adjust the benchmark file (i.e. Benchmark/primality_testing.json) and adjust the hyperparameter (max_length, max_new_token, temperature, num_beams, do_sample). Then run the script.

For example:

```train
python3 phi_1B.py
```

## Evaluation

To evaluate the result:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 