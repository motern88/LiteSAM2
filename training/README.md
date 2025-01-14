# Training Code for SAM 2

This folder contains the training code for SAM 2, a foundation model for promptable visual segmentation in images and videos. 
The code allows users to train and fine-tune SAM 2 on their own datasets (image, video, or both).

## Structure

The training code is organized into the following subfolders:

* `dataset`: This folder contains image and video dataset and dataloader classes as well as their transforms.
* `model`: This folder contains the main model class (`SAM2Train`) for training/fine-tuning. `SAM2Train` inherits from `SAM2Base` model and provides functions to enable training or fine-tuning SAM 2. It also accepts all training-time parameters used for simulating user prompts (e.g. iterative point sampling).
* `utils`: This folder contains training utils such as loggers and distributed training utils.
* `scripts`: This folder contains the script to extract the frames of SA-V dataset to be used in training.
* `loss_fns.py`: This file has the main loss class (`MultiStepMultiMasksAndIous`) used for training.
* `optimizer.py`:  This file contains all optimizer utils that support arbitrary schedulers.
* `trainer.py`: This file contains the `Trainer` class that accepts all the `Hydra` configurable modules (model, optimizer, datasets, etc..) and implements the main train/eval loop.
* `train.py`: This script is used to launch training jobs. It supports single and multi-node jobs. For usage, please check the [Getting Started](README.md#getting-started) section or run `python training/train.py -h`

## Getting Started

To get started with the training code, we provide a simple example to fine-tune our checkpoints on [MOSE](https://henghuiding.github.io/MOSE/) dataset, which can be extended to your custom datasets.

#### Requirements:
- We assume training on A100 GPUs with **80 GB** of memory.
- Download the MOSE dataset using one of the provided links from [here](https://github.com/henghuiding/MOSE-api?tab=readme-ov-file#download).

#### Steps to fine-tune on MOSE:
- Install the packages required for training by running `pip install -e ".[dev]"`.
- Set the paths for MOSE dataset in `configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`.
    ```yaml
    dataset:
        # PATHS to Dataset
        img_folder: null # PATH to MOSE JPEGImages folder
        gt_folder: null # PATH to MOSE Annotations folder
        file_list_txt: null # Optional PATH to filelist containing a subset of videos to be used for training
    ```
- To fine-tune the base model on MOSE using 8 GPUs, run 

    ```python
    python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 0 \
        --num-gpus 8
    ```

    We also support multi-node training on a cluster using [SLURM](https://slurm.schedmd.com/documentation.html), for example, you can train on 2 nodes by running

    ```python
    python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 1 \
        --num-gpus 8 \
        --num-nodes 2
        --partition $PARTITION \
        --qos $QOS \
        --account $ACCOUNT
    ```
    where partition, qos, and account are optional and depend on your SLURM configuration.
    By default, the checkpoint and logs will be saved under `sam2_logs` directory in the root of the repo. Alternatively, you can set the experiment log directory in the config file as follows:
  
    ```yaml
      experiment_log_dir: null # Path to log directory, defaults to ./sam2_logs/${config_name}
    ```
    The training losses can be monitored using `tensorboard` logs stored under `tensorboard/` in the experiment log directory. We also provide a sample validation [split]( ../training/assets/MOSE_sample_val_list.txt) for evaluation purposes. To generate predictions, follow this [guide](../tools/README.md) on how to use our `vos_inference.py` script. After generating the predictions, you can run the `sav_evaluator.py` as detailed [here](../sav_dataset/README.md#sa-v-val-and-test-evaluation). The expected MOSE J&F after fine-tuning the Base plus model is 79.4.
    
    
    After training/fine-tuning, you can then use the new checkpoint (saved in `checkpoints/` in the experiment log directory) similar to SAM 2 released checkpoints (as illustrated [here](../README.md#image-prediction)).
## Training on images and videos
The code supports training on images and videos (similar to how SAM 2 is trained). We provide classes for loading SA-1B as a sample image dataset, SA-V as a sample video dataset, as well as any DAVIS-style video dataset (e.g. MOSE). Note that to train on SA-V, you must first extract all videos to JPEG frames using the provided extraction [script](./scripts/sav_frame_extraction_submitit.py). Below is an example of how to setup the datasets in your config to train on a mix of image and video datasets:

```yaml
data:
  train:
    _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset 
    phases_per_epoch: ${phases_per_epoch} # Chunks a single epoch into smaller phases
    batch_sizes: # List of batch sizes corresponding to each dataset
    - ${bs1} # Batch size of dataset 1
    - ${bs2} # Batch size of dataset 2
    datasets:
    # SA1B as an example of an image dataset
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist} # Optional
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 1
        max_num_objects: ${max_num_objects_per_image}
      transforms: ${image_transforms}
    # SA-V as an example of a video dataset
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.JSONRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist} # Optional
        ann_every: 4
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 8 # Number of frames per video
        max_num_objects: ${max_num_objects_per_video}
        reverse_time_prob: ${reverse_time_prob} # probability to reverse video
      transforms: ${video_transforms}
    shuffle: True
    num_workers: ${num_train_workers}
    pin_memory: True
    drop_last: True
    collate_fn:
    _target_: training.utils.data_utils.collate_fn
    _partial_: true
    dict_key: all
```

# SAM 2 训练代码

此文件夹包含 SAM 2 的训练代码，这是一个用于图像和视频可提示视觉分割的基础模型。代码允许用户在自己的数据集（图像、视频或两者）上训练和微调 SAM 2。

## 结构

训练代码组织成以下子文件夹：

* `dataset`：包含图像和视频数据集及其 dataloader 类以及数据增强。
* `model`：包含用于训练/微调的主要模型类 (`SAM2Train`)，该类继承自 `SAM2Base` 模型，并提供了启用训练或微调 SAM 2 的功能。它还接受所有在训练时使用的参数，用于模拟用户提示（例如迭代点采样）。
* `utils`：包含训练的工具类，如日志记录和分布式训练工具。
* `scripts`：包含用于提取 SA-V 数据集帧的脚本。
* `loss_fns.py`：包含用于训练的主要损失类 (`MultiStepMultiMasksAndIous`)。
* `optimizer.py`：包含支持任意调度器的优化器工具。
* `trainer.py`：包含 `Trainer` 类，该类接受所有由 `Hydra` 配置的模块（模型、优化器、数据集等），并实现了主要的训练/评估循环。
* `train.py`：此脚本用于启动训练任务，支持单节点和多节点任务。有关使用方法，请查阅 [Getting Started](README.md#getting-started) 部分或运行 `python training/train.py -h`。

## 快速开始

为了快速开始训练代码，我们提供了一个简单的示例，说明如何在 [MOSE](https://henghuiding.github.io/MOSE/) 数据集上微调我们的模型，这个过程可以扩展到你自己的自定义数据集。

#### 环境要求：
- 我们假设在 **80 GB** 内存的 A100 GPU 上进行训练。
- 使用 [此链接](https://github.com/henghuiding/MOSE-api?tab=readme-ov-file#download) 下载 MOSE 数据集。

#### 在 MOSE 上微调的步骤：
- 通过运行 `pip install -e ".[dev]"` 安装训练所需的包。
- 在 `configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml` 中设置 MOSE 数据集的路径。
    ```yaml
    dataset:
        # 数据集的路径
        img_folder: null # MOSE JPEGImages 文件夹的路径
        gt_folder: null # MOSE 注释文件夹的路径
        file_list_txt: null # 可选：包含要用于训练的部分视频的文件列表路径
    ```
- 使用 8 个 GPU 在 MOSE 上微调基础模型，运行：

    ```python
    python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 0 \
        --num-gpus 8
    ```

    我们也支持在集群上进行多节点训练，可以使用 [SLURM](https://slurm.schedmd.com/documentation.html) 进行训练，例如，使用 2 个节点进行训练：

    ```python
    python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 1 \
        --num-gpus 8 \
        --num-nodes 2 \
        --partition $PARTITION \
        --qos $QOS \
        --account $ACCOUNT
    ```

    其中 partition、qos 和 account 是可选的，具体取决于您的 SLURM 配置。
    默认情况下，检查点和日志将保存在根目录下的 `sam2_logs` 目录中。或者，您可以在配置文件中设置实验日志目录，如下所示：

    ```yaml
      experiment_log_dir: null # 日志目录路径，默认为 ./sam2_logs/${config_name}
    ```

    可以使用 `tensorboard` 监控训练损失，日志存储在实验日志目录下的 `tensorboard/` 中。我们还提供了一个示例验证集 [split]( ../training/assets/MOSE_sample_val_list.txt) 供评估使用。要生成预测，请参阅 [指南](../tools/README.md)，了解如何使用我们的 `vos_inference.py` 脚本。在生成预测后，可以运行 `sav_evaluator.py`，详情请查看 [这里](../sav_dataset/README.md#sa-v-val-and-test-evaluation)。微调后，预期的 MOSE J&F 值为 79.4。

    训练/微调后，您可以像使用 SAM 2 发布的检查点一样使用新的检查点（保存在 `checkpoints/` 目录中，位于实验日志目录下），具体操作可以参考 [此处](../README.md#image-prediction)。

## 在图像和视频上进行训练
该代码支持在图像和视频上进行训练（与 SAM 2 的训练方式类似）。我们提供了加载 SA-1B 作为图像数据集示例，SA-V 作为视频数据集示例，以及任何 DAVIS 风格的视频数据集（例如 MOSE）。请注意，要在 SA-V 上训练，您必须首先使用提供的提取 [脚本](./scripts/sav_frame_extraction_submitit.py) 将所有视频提取为 JPEG 帧。以下是如何在配置中设置数据集，以便在图像和视频数据集的混合上进行训练的示例：

```yaml
data:
  train:
    _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset 
    phases_per_epoch: ${phases_per_epoch} # 将单个 epoch 切分为更小的阶段
    batch_sizes: # 每个数据集对应的批次大小列表
    - ${bs1} # 数据集 1 的批次大小
    - ${bs2} # 数据集 2 的批次大小
    datasets:
    # SA1B 作为图像数据集的示例
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist} # 可选
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 1
        max_num_objects: ${max_num_objects_per_image}
      transforms: ${image_transforms}
    # SA-V 作为视频数据集的示例
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.JSONRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist} # 可选
        ann_every: 4
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 8 # 每个视频的帧数
        max_num_objects: ${max_num_objects_per_video}
        reverse_time_prob: ${reverse_time_prob} # 反转视频的概率
      transforms: ${video_transforms}
    shuffle: True
    num_workers: ${num_train_workers}
    pin_memory: True
    drop_last: True
    collate_fn:
    _target_: training.utils.data_utils.collate_fn
    _partial_: true
    dict_key: all
```