[![CodeQL](https://github.com/smazcw3/LLM_world/actions/workflows/codeql.yml/badge.svg)](https://github.com/smazcw3/LLM_world/actions/workflows/codeql.yml)

# LLM_world
This repository is for researching on Large Language Models (LLM).

Important notes
---------------
+ Each parameter in an LLM takes up 4 bytes (i.e. it is a 32-bit precision number) just to store the model. 1B parameters will take about 4GB of GPU RAM.
+ GPT-3 has 175B parameters => 700 GB of GPU RAM just to store the model.
+ Converting to 16 bit half precision floating point (FP16) requires 2 bytes per parameter => storage requirements gets halved.  
+ Converting to 8 bit (FP8) requires 1 byte per parameter, however each parameter is an integer => storage requirements gets 1/4.
+ Training a model can use 20 byte/parameter to store it (optimizer states, gradients, activations and any temporary variables).

Some of the Hugging-face based notebooks are below:

| Notebook name | Colab links |
| --- | --- |
| `Accessing pre-trained models` | https://colab.research.google.com/drive/1XxuUY3lV7EtKxxThFcxrp2hpxADnGUWg?usp=sharing |
| `How To Use Transformers` | https://colab.research.google.com/drive/1Rv9xVchNq_YTPZ1OwruTS2-WuG-yxLqa?usp=sharing |
| `Working with Datasets` | https://colab.research.google.com/drive/1Yqn0VIwyoo8ISKJ51KMRvKl-Yq5HufxJ?usp=sharing |
| `Creating a Tokenizer` | https://colab.research.google.com/drive/1uJUBE2HWrQmfIXnZ9-9X1mYIZY8NjYNa?usp=sharing |
| `Full Fine Tuning for Summarization` | https://colab.research.google.com/drive/1QhkLc4UQ0t0_y3nk3eSVoUQ-99_Tiv_I?usp=sharing |
| `PEFT_For_Summarization` | https://colab.research.google.com/drive/1dBrN3BSFoKBY_nqqDd9dQCLr2CKz3A7w?usp=sharing |