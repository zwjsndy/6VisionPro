# 6Vision

**6Vision: Image-encoding-based IPv6 Target Generation in Few-seed Scenarios**





## Environmental requirement

- pytorch 2.9.1 (CUDA 12.8)
- torchvision 0.24.1
- Python 3.10.19
- SMap (more information at https://github.com/AddrMiner/smap)
- Additional packages: numpy, matplotlib, scikit-learn, scipy, pandas, captum, pyasn, seaborn, tqdm, pillow (see `requirements.txt`)

## SMap

SMap is a high-performance programming framework designed for large-scale network measurement. Its core functionality is similar to that of ZMap, but it has been deeply optimized in terms of scalability, engineering standards, and customization capabilities. SMap is developed using the modern programming language Rust, offering advantages such as high performance, strong stability, and clean, maintainable code. See usage and more information at https://github.com/AddrMiner/smap.

## Suggested Working Directory Structure

We recommend organizing the structure of the working directory for 6Vision as follows:

```text
6Vision-main/
├── model/                     # for trained and fine-tuned models
│   ├── trained_models.pth
│   └── finetuned_models.pth 
├── temp/
│   ├── generated_addresses.txt # for probing in smap
│   ├── generated_logs.jsonl. # for retrain
│   ├── output_from_smap.txt  # active addresses with tags (with aliased)
│   ├── extract_address.py  # extract raw addresses from smap output files
│   ├── raw_active_address.txt  # extracted raw active (with aliased) addresses
│   ├── aliased_from_smap.txt  # scanned aliased prefixes & addresses from smap
│   ├── filter_aliased.py  # separate active & aliased addresses for retrain
│   ├── active_finals.txt  # pure active addresses for retrain (no aliased)
│   ├── aliased_finals.txt  # pure aliased addresses for retrain
│   └── combine_results.txt  # to prevent identical results in future rounds
├── my_seeds.txt  # your initial seed file
├── label.txt  # labels for my_seeds.txt
├── cluster.py
├── train_gatedpixelcnn.py
├── generate.py
├── retrain.py
├── requirements.txt
└── README.md
```

**NOTE**: Please run `extract_address.py` and `filter_aliased.py` under `./temp/` , otherwise files will not be correctly read.
## Running Instructions

### Input Seed Files

The input is a `.txt` file containing one IPv6 seed address per line. Place it in the working directory as `my_seeds.txt`(or other names).

For example:

```
2a01:1111::1
2a01:1111::2
2a01:1112::1
2a01:1112::3
2a01:1113::2
```

### Steps

**1. Clustering**

Run `cluster.py` to cluster the seed addresses into groups using a VAE convolutional autoencoder + hierarchical clustering. The output is a `.txt` file assigning each seed to a cluster.

bash

```bash
python cluster_gc.py --seed_file ./my_seeds.txt --label_file ./my_labels.txt --k 6
```

| Argument       | Default       | Description             |
| -------------- | ------------- | ----------------------- |
| `--seed_file`  | _(required)_  | Path to your seed file. |
| `--label_file` | `./label.txt` | Output label file.      |
| `--k`          | 6             | Number of clusters.     |

**2. Model Training**

Run `train_gatedpixelcnn.py` to train one independent GatedPixelCNN model per cluster. Models are saved to directory `./model/`.

bash

```bash
python train_gatedpixelcnn.py --seed_file ./my_seeds.txt --label_file ./my_labels.txt
```

| Argument       | Default       | Description                                                             |
| -------------- | ------------- | ----------------------------------------------------------------------- |
| `--k`          | 6             | Number of clusters.                                                     |
| `--stack_size` | 6             | Number of rows in the input stack (e.g., 6 for 5 context+1 generation). |
| `--seed_file`  | _(required)_  | Path to your seed file.                                                 |
| `--label_file` | `./label.txt` | Output label file.                                                      |

**3. Address Generation**

Run `generate.py` to generate candidate IPv6 addresses using the trained models. All clusters are processed in parallel across available GPUs. The generation results will be saved in files with names such as `res_{cluster_id}_with_1_6_128.txt`, and the log output during generation will be stored as `generated_log_res_{cluster_id}_with_1_6_128.jsonl`. Generation results and output logs will be save under directory `./temp/`. Use the argument `exclude_file` after the initial round of retraining to prevent generating identical address in future rounds.

bash

```bash
# Round 1: loads cluster_0_with_1_6_128.pth (default, no suffix)
python generate.py --budget 200000 --seed_file ./your_seeds.txt

# Round 2: loads cluster_0_RL_finetuned_with_1_6_128.pth
python generate.py --budget 200000 --seed_file ./your_seeds.txt --model_name_suffix RL_finetuned --exclude_file ./temp/combined_res_k6_s6.txt
```

| Argument               | Default       | Description                                                                                                                                                       |
| ---------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--budget`             | _(required)_  | Total generation budget across all clusters.                                                                                                                      |
| `--k`                  | 6             | Number of clusters.                                                                                                                                               |
| `--budget_per_cluster` | auto          | Budget per cluster (overrides even split of `--budget`).                                                                                                          |
| `--stack_size`         | 6             | Stack size matching the trained model.                                                                                                                            |
| `--exclude_file`       | None          | One or more `.txt` files of previously generated addresses under `./temp/` to exclude from output.                                                                |
| `--model_name_prefix`  | `cluster`     | Model filename prefix for Round 1 (default: `cluster`, this will load `cluster_{i}_with_1_6_128.pth`).                                                            |
| `--model_name_suffix`  | none          | Optional suffix inserted after cluster id for Round 2+ (default: none). E.g. when enter `RL_finetuned`, this will load `cluster_0_RL_finetuned_with_1_6_128.pth`. |
| `--seed_file`          | _(required)_  | Path to your seed file.                                                                                                                                           |
| `--label_file`         | `./label.txt` | Output label file.                                                                                                                                                |

**4. Address Probing**

This step requires a machine with SMap installed (check usage and more information at https://github.com/AddrMiner/smap). Enter the following command on the machine with SMap installed and obtain the active address results (with aliased addresses included).

bash

```bash
smap -m f6 -b 10m -f res_{cluster_id}_with_1_6_128.txt --output_file_v6 output_{i}.txt
```

Scanned active address will be stored in your designated file output path, such as `output_{i}.txt`, which will be used during active address extraction and aliased address detection. Please move these scanned active address files to the `./temp/` folder. 

Note: Raw results scanned active results will be stored per row with columns names as the first row:

```txt
source_addr,outer_source_addr,icmp_type,icmp_code,identifier,sequence,classification
2804:14c:390::c,,129,0,46281,0,echo_reply
```

To extract all raw addresses, please run `extract_address.py` under `./temp/`.

bash

```bash
python extract_address.py --input_file_path ./output --output_file_path ./active_output
```

| Argument             | Default      | Description                                                                                                                                      |
| -------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--input_file_path`  | _(required)_ | File paths for files from active address probing in SMap (e.g. `./output`, this will read output files `output_{i}.txt` files under `./temp/` ). |
| `--output_file_path` | _(required)_ | Output file paths for extracted addresses (e.g. `./active_output`, this will generate `active_output_{i}.txt` files under `./temp/` ).           |
| `--k`                | 6            | Number of clusters.                                                                                                                              |

**5. Aliased detection**

After acquiring all the raw active output addresses generated, run the following command on the machine with SMap installed to probe aliased addresses in the raw active address file.

bash

```bash
smap -m ac6 -f active_output_{i}.txt -a prefix_len=64 -a rand_addr_len=16 -a alia_ratio=0.4 -a output_alia_addrs=true -b 5m -a prefixes_len_per_batch=1000000 --output_file_v6 aliased_res_{i}.txt
```

You will obtain `.txt` files containing aliased prefixes and addresses. Move them to `./temp/`. Then, run the `filter_aliased.py` under `./temp/` to separate non-aliased addresses and aliased addresses to obtain results for the reinforcement learning process.

bash

```bash
python filter_aliased.py --aliased_file_path aliased_res --active_file_path active_output --active_final_file_path active_final --aliased_final_file_path aliased_final
```

| Argument                    | Default      | Description                                                                                                          |
| --------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `--aliased_file_path`       | _(required)_ | Aliased output file paths from SMap (e.g. `aliased_res`, this will read `aliased_res_{i}.txt`).                      |
| `--active_file_path`        | _(required)_ | Active addresses (with aliased) file paths from SMap (e.g. `active_output`, this will read `active_output_{i}.txt`). |
| `--active_final_file_path`  | _(required)_ | Output final active file paths (e.g. 'active\_final', this will produce `active_final_{i}.txt`).                     |
| `--aliased_final_file_path` | _(required)_ | Output final aliased file paths (e.g. 'aliased\_final', this will produce `aliased_final_{i}.txt`).                  |
| `--k`                       | 6            | Number of clusters.                                                                                                  |
At the end of this step, all necessary files for the RL retraining process are obtained: the label file (step 1), the original to-be-tuned model files (step 2), the output logs when generating a round of addresses (step 3), the final aliased address files (step 5) and the final active address files (step 5).

**6. Fine-Tuning based on Reinforcement Learning**

Run `retrain.py` to fine-tune all cluster models using REINFORCE with the probing results in previous steps (active/aliased feedback). All clusters are processed sequentially.

bash

```bash
python retrain.py
```

| Argument                    | Default             | Description                                                                                                                                                     |
| --------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--k`                       | 6                   | Number of clusters.                                                                                                                                             |
| `--stack_size`              | 6                   | Stack size matching the trained model.                                                                                                                          |
| `--model_name_prefix`       | `cluster`           | Prefix for input model files (e.g. `cluster`, this will read `cluster_{i}_with_1_6_128.pth` under `./model/`).                                                  |
| `--log_name_prefix`         | `generated_log_res` | Prefix for generation log files (e.g. `generated_log_res`, this will read `generated_log_res_with_1_6_128.jsonl` under `./temp/`).                              |
| `--model_save_name`         | `RL_finetuned`      | Prefix for saved fine-tuned model files (e.g. `RL_finetuned`, this will save the updated model as `cluster_{i}_RL_finetuned_with_1_6_128.pth`under `./model/`). |
| `--active_final_file_path`  | `active_final`      | Prefix for final active address files (e.g. `active_final`, this will read `active_final_{i}.txt` under `./temp/`).                                             |
| `--aliased_final_file_path` | `aliased_final`     | Prefix for final aliased address files (e.g. `aliased_final`, this will read `aliased_final_{i}.txt` under `./temp/`).                                          |
| `--seed_prefixes`           | optional            | If you wish to add prefix restraint during the retrain, enter this argument for additional penalty on out-of-prefixes addresses.                                |

**7. Iterative Refinement**

To perform additional rounds of fine-tuning, repeat steps 3–6. Use the `--exclude_file` argument in step 3 to avoid regenerating addresses from previous rounds, and update the `--model_name_prefix` / `--log_name_prefix` / `--model_save_name` / `active_final_file_path` / `aliased_final_file_path` in step 6 to point to the latest files. Each round improves generation quality by incorporating probing feedback. If you wish to retrain for multiple rounds, please make sure to input correct files to the codes.

**NOTE**: when generating a new round of addresses, be sure to exclude addresses that were generating in previous rounds. To do this, please run the `combine_results.py` under `./temp/` to combine results from clusters in previous rounds into one file and exclude these addresses in the argument `--exlude_file` in step 3.

bash

```bash
python combine_results.py previous_generated_output_files(separate by space)
```

| Argument        | Default        | Description                                                                                                               |
| --------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--k`           | 6              | Number of clusters.                                                                                                       |
| `--stack_size`  | 6              | Stack size to identify correct files to combine.                                                                          |
| `--input_dir`   | `./`           | Directory containing to-be-combined files.                                                                                |
| `--result_name` | `res`          | Results filename: e.g. `res` for first-round, this will combine `k` results that are named as `res_{i}_with_1_6_128.txt`. |
| `--output`      | auto-generated | Output file path. Auto-generated output for `res` results filename is `combined_res_k{K_VALUE}_s{STACK_SIZE}.txt`         |
# 6VisionPro
