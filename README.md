# Benchmarking Direct Preference Optimization for Medical Vision-Language Models

This repository contains the official implementation for benchmarking various Direct Preference Optimization (DPO) methodologies on Medical Large Vision-Language Models (VLMs).

## 🛠️ Environment Setup

To set up the environment and install all necessary dependencies, run the following command in the root directory of the project:

```bash
pip install -e .

```

## 🚀 Models and Methodologies

We implemented and evaluated multiple preference optimization algorithms targeting two prominent Medical VLMs.

### Target Medical VLMs

* **LLaVA-Med** ([Paper](https://arxiv.org/abs/2306.00890))
* **HuatuoGPT-Vision-7B** ([Paper](https://arxiv.org/abs/2406.19280))

### Implemented Preference Optimization Methods

* **DPO**: Direct Preference Optimization ([Paper](https://arxiv.org/abs/2305.18290))
* **DPO+NLL**: DPO with Negative Log-Likelihood ([Paper](https://arxiv.org/abs/2404.19733))
* **COPO**: Conditional Preference Optimization ([Paper](https://arxiv.org/abs/2406.11839))
* **mDPO** ([Paper](https://arxiv.org/abs/2406.11839))
* **MMedPO** ([Paper](https://arxiv.org/abs/2412.06141))

## 🏃‍♂️ Training Scripts

We provide ready-to-use shell scripts to easily train your desired model using the preferred optimization method.

The scripts follow a unified naming convention:
`train_[Training Method]__[Model Name]-IT.sh`

> **Note:** Before running the scripts, make sure to replace `<YOUR_WORKSPACE_PATH>` and `<YOUR_WANDB_API_KEY>` inside the shell files with your actual local paths and credentials.

### Training Configuration Overview

| Script Name | Target Model | DPO Methodology |
| --- | --- | --- |
| `train_dpo__llava-med-IT.sh` | LLaVA-Med | DPO |
| `train_dpo+nll__llava-med-IT.sh` | LLaVA-Med | DPO+NLL |
| `train_copo__llava-med-IT.sh` | LLaVA-Med | COPO |
| `train_mDPO__llava-med-IT.sh` | LLaVA-Med | mDPO |
| `train_MMedPO__llava-med-IT.sh` | LLaVA-Med | MMedPO |
| `train_dpo__huatuo-IT.sh` | HuatuoGPT-Vision-7B | DPO |
| `train_dpo+nll__huatuo-IT.sh` | HuatuoGPT-Vision-7B | DPO+NLL |
| `train_copo__huatuo-IT.sh` | HuatuoGPT-Vision-7B | COPO |
| `train_mDPO__huatuo-IT.sh` | HuatuoGPT-Vision-7B | mDPO |
| `train_MMedPO__huatuo-IT.sh` | HuatuoGPT-Vision-7B | MMedPO |

## 📂 Data Samples

You can explore the expected structure and formatting of our training datasets by checking the sample files located in the following directory:

```bash
./data/benchmark_question_file

```

## 🙏 Acknowledgements

This project is built upon the foundational codebase of [HA-DPO](https://github.com/opendatalab/HA-DPO). We sincerely thank the authors for their fantastic work and for open-sourcing their code to the community.

## 📝 Citation

If you find this repository or our benchmarking results useful in your research, please consider citing our paper:

```bibtex
@inproceedings{kim2026benchmarking,
  title={Benchmarking Direct Preference Optimization for Medical Large Vision-Language Models},
  author={Kim, Dain and Lee, Jiwoo and Yun, Jaehoon and Koo, Yong Hoe and Chen, Qingyu and Kim, Hyunjae and Kang, Jaewoo},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2026},
  year={2026},
  url={https://arxiv.org/abs/2601.17918}
}

```