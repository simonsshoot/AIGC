# run_detector_MFCL.py

# coding=utf-8
# ...（版权声明和其他注释）...
import os
import torch
import argparse
import logging
import random
import wandb
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
from datetime import datetime
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from torch.optim import AdamW

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
)


try:
    from apex import amp
except ImportError:
    amp = None
from util import glue_compute_metrics as compute_metrics
from util import (
    glue_convert_examples_to_features as convert_examples_to_features,
)
from util import glue_output_modes as output_modes
from util import glue_processors as processors
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
except ImportError:
    ray = None

# 自定义模块
from modeling_roberta_MFCL import (
    RobertaForGraphBasedSequenceClassification_CL,
    RobertaForContrastiveSequenceClassification,
    RobertaForGraphBasedSequenceClassification_MBCL,
    RobertaForGraphBasedSequenceClassification_RFCL,
)
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from datetime import datetime


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    default=os.path.join(os.getcwd(), "data"),
    type=str,
    help="The input data dir.",
)
parser.add_argument(
    "--model_type",
    default="roberta",
    type=str,
    help="Base model for CoCo",
)
parser.add_argument(
    "--model_name_or_path",
    default="/home/yyh/yyh/PythonProject/Coh-MGT-Detection/preprocess/roberta-base",  # 默认本地路径
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--task_name",
    default="deepfake",
    type=str,
)
# parser.add_argument(
#     "--output_dir",
#     default=os.path.join(os.getcwd(), "gpt2_1000_test"),
#     type=str,
#     required=True,
#     help="The output directory where the model predictions and checkpoints will be written.",
# )
# 获取当前时间并格式化
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 使用 dataset_name 生成 output_dir 路径
parser.add_argument(
    "--output_dir",
    default=os.path.join(os.getcwd(), "test", f"{parser.get_default('dataset_name')}_{current_time}"),
    type=str,
    required=False,
    help="The output directory where the model predictions and checkpoints will be written.",
)

parser.add_argument(
    "--config_name",
    default="",
    type=str,
    help="Pretrained config name or path if not the same as model_name",
)
parser.add_argument(
    "--train_file", default="p\=0.96.jsonl", type=str, help="training file"
)
parser.add_argument(
    "--dev_file", default="p\=0.96.jsonl", type=str, help="training file"
)
parser.add_argument(
    "--test_file", default="p\=0.96.jsonl", type=str, help="training file"
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--do_train", 
    default=True, 
    help="Whether to run training.")
parser.add_argument(
    "--do_eval", 
    default=True, 
    help="Whether to run eval on the dev set."
)
parser.add_argument(
    "--do_test", 
    default=True, 
    help="Whether to run test on the dev set."
)
parser.add_argument(
    "--evaluate_during_training",
    action="store_true",
    help="Run evaluation during training at each logging step.",
)
parser.add_argument(
    "--do_lower_case",
    action="store_true",
    help="Set this flag if you are using an uncased model.",
)
parser.add_argument(
    "--per_gpu_train_batch_size",
    default=16,
    type=int,
    help="Batch size per GPU/CPU for training.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=16,
    type=int,
    help="Batch size per GPU/CPU for evaluation.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="The initial learning rate for Adam.",
)
parser.add_argument(
    "--weight_decay", 
    default=0.01, 
    type=float, 
    help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", 
    default=1e-8, 
    type=float, 
    help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--max_grad_norm", 
    default=1.0, 
    type=float, 
    help="Max gradient norm."
)
parser.add_argument(
    "--num_train_epochs",
    default=10,
    type=float,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument(
    "--warmup_steps", 
    default=0, 
    type=int, 
    help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--logging_steps", 
    type=int, 
    default=125, 
    help="Interval certain steps to log."
)
parser.add_argument(
    "--save_steps", 
    type=int, 
    default=500, 
    help="Interval certain steps to save checkpoint."
)
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument(
    "--no_cuda", 
    action="store_true", 
    help="Avoid using CUDA when available"
)
parser.add_argument(
    "--overwrite_output_dir",
    type=bool,
    default=True,
    help="Overwrite the content of the output directory",
)
parser.add_argument(
    "--overwrite_cache",
    default=True,
    help="Overwrite the cached training and evaluation sets",
)
parser.add_argument(
    "--seed", 
    type=int, 
    default=0, 
    help="Random seed."
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument(
    "--local_rank", 
    type=int, 
    default=-1, 
    help="For distributed training: local_rank"
)
parser.add_argument(
    "--server_ip", 
    type=str, 
    default="", 
    help="For distant debugging."
)
parser.add_argument(
    "--server_port", 
    type=str, 
    default="", 
    help="For distant debugging."
)
parser.add_argument(
    "--max_nodes_num", 
    type=int, 
    default=150, 
    help="Maximum of number of nodes when input."
)
parser.add_argument(
    "--max_sentences", 
    type=int, 
    default=45, 
    help="Maximum of number of sentences when input."
)
parser.add_argument(
    "--max_sen_replen",
    type=int,
    default=128,
    help="Maximum of length of sentences representation (after relu).",
)
parser.add_argument(
    "--attention_maxscore",
    type=int,
    default=16,
    help="Weight of the max similarity score inside self-attention.",
)
parser.add_argument(
    "--loss_type",
    default="contrastive",
    type=str,
    help="Loss Type, include: normal, scl, mbcl, rfcl. rfcl is the complete version of CoCo, normal is the baseline.",
)
parser.add_argument(
    "--gcn_layer",
    default=2,
    type=int,
    help="Number of layers of GAT, recommand 2.",
)
parser.add_argument(
    "--dataset_name",
    default="gpt3.5_mixed",
    type=str,
    help="Name of the dataset, if blank will use Grover dataset",
)
parser.add_argument(
    "--do_ray",
    default=False,
    type=bool,
    help="Searching hyperparameter by Ray Tune or not",
)
parser.add_argument(
    "--with_relation",
    default=2,
    type=int,
    help="number of relation in Relation-GCN, >=2 for multi-relation, and =0 for the vanilla GCN.",
)
parser.add_argument(
    "--wandb_note",
    default="CoCo_rf",
    type=str,
    help="To describe the name of Wandb record.",
)

# 添加对比学习的命令行参数
parser.add_argument(
    "--contrastive_weight",
    type=float,
    default=0.1,
    help="Weight for the contrastive loss.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.07,
    help="Temperature for contrastive loss.",
)
parser.add_argument(
    "--use_triplet_loss",
    action="store_true",
    help="Whether to use triplet contrastive loss.",
)
parser.add_argument(
    "--triplet_margin",
    type=float,
    default=1.0,
    help="Margin for triplet loss.",
)
args = parser.parse_args()
def generate_shaped_nodes_mask(nodes, max_seq_length, max_nodes_num):
    nodes_mask = np.zeros(shape=(max_nodes_num, max_seq_length))
    nodes_num = min(len(nodes), max_nodes_num)

    for i in range(nodes_num):
        span = nodes[i]
        if span[0] != -1:
            if span[0] < max_seq_length - 1:
                end_pos = (
                    span[1] if span[1] < max_seq_length - 1 else max_seq_length - 1
                )
                nodes_mask[i, span[0] + 1 : end_pos + 1] = 1
            else:
                continue
    return nodes_mask, nodes_num


def generate_shaped_edge_mask(adj_metric, nodes_num, max_nodes_num, relation_n):
    if nodes_num != 0:
        if relation_n != 0:
            new_adj_metric = np.zeros(shape=(relation_n, max_nodes_num, max_nodes_num))
            for i in range(relation_n):
                new_adj_metric[i][:nodes_num, :nodes_num] = adj_metric[i][
                    :nodes_num, :nodes_num
                ]
        else:
            new_adj_metric = np.zeros(shape=(max_nodes_num, max_nodes_num))
            new_adj_metric[:nodes_num, :nodes_num] = adj_metric[:nodes_num, :nodes_num]
    return new_adj_metric
# 定义 ContrastiveDataset 类
class ContrastiveDataset(Dataset):
    """
    自定义Dataset，每次返回一个锚点样本、一个随机攻击样本、
    一个正样本和一个负样本，用于计算两种对比损失。
    假设数据集按每12个为一组排列：
    - 第1个为锚点样本
    - 第2到第10个为第一种对比损失的攻击样本
    - 第11个为第二种对比损失的正样本
    - 第12个为第二种对比损失的负样本
    """

    def __init__(self, tensor_dataset, group_size=12, seed=0):
        self.tensor_dataset = tensor_dataset
        self.group_size = group_size
        self.num_groups = len(tensor_dataset) // group_size
        self.random = random.Random(seed)

    def __len__(self):
        return self.num_groups

    def __getitem__(self, idx):
        start = idx * self.group_size
        end = start + self.group_size
        group = [self.tensor_dataset[i] for i in range(start, end)]
        
        anchor = group[0]  # 锚点样本
        contrastive_samples = group[1:10]  # 第2到第10个为第一种对比损失的攻击样本
        positive = group[10]  # 第11个为第二种对比损失的正样本
        negative = group[11]  # 第12个为第二种对比损失的负样本

        # 随机选择一个攻击样本用于第一种对比损失
        selected_attack = self.random.choice(contrastive_samples)

        return anchor, selected_attack, positive, negative

# 定义三元组对比损失函数
def compute_triplet_contrastive_loss(anchor_rep, positive_rep, negative_rep, margin=1.0):
    """
    计算三元组对比损失。
    """
    # L2距离
    distance_positive = F.pairwise_distance(anchor_rep, positive_rep, p=2)
    distance_negative = F.pairwise_distance(anchor_rep, negative_rep, p=2)
    
    # 三元组损失
    loss = F.relu(distance_positive - distance_negative + margin)
    
    return loss.mean()

# 定义普通对比损失函数
def compute_contrastive_loss(original_rep, attack_rep, temperature=0.2):
    """
    计算普通对比损失
    """
    original_rep = F.normalize(original_rep, dim=1)
    attack_rep = F.normalize(attack_rep, dim=1)

    # 计算余弦相似度
    logits = torch.matmul(original_rep, attack_rep.t()) / temperature

    # 目标标签为对角线（即每个原始样本对应一个攻击样本）
    labels = torch.arange(original_rep.size(0)).to(original_rep.device)

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels)

    return loss

# 定义模型训练函数
def train(args, train_dataset, model, tokenizer):
    """Train the model with multiple contrastive learning objectives"""
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Total Params:", total_params)
    print("Total Trainable Params:", total_trainable_params)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else torch.utils.data.distributed.DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )

    # Check if saved optimizer or scheduler states exist
    if (
        os.path.exists(args.model_name_or_path)
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        if amp is None:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    best_acc, best_f1 = 0.0, 0.0
    global_step, epochs_trained, steps_trained_in_current_epoch = 0, 0, 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args.seed)  # 修改这里
    max_acc, max_acc_f1, max_f1, max_f1_acc = 0.0, 0.0, 0.0, 0.0
    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # 每个batch包含(anchor, attack, positive, negative)四个样本
            anchor_batch, attack_batch, positive_batch, negative_batch = batch

            model.train()
            anchor_batch = tuple(t.to(args.device) for t in anchor_batch)
            attack_batch = tuple(t.to(args.device) for t in attack_batch)
            positive_batch = tuple(t.to(args.device) for t in positive_batch)
            negative_batch = tuple(t.to(args.device) for t in negative_batch)

            # 准备锚点样本的输入
            anchor_inputs = {
                "input_ids": anchor_batch[0],
                "attention_mask": anchor_batch[1],
                "labels": anchor_batch[3],
                "nodes_index_mask": anchor_batch[4],
                "adj_metric": anchor_batch[5],
                "node_mask": anchor_batch[6],
                "sen2node": anchor_batch[7],
                "sentence_mask": anchor_batch[8],
                "sentence_length": anchor_batch[9],
                "batch_id": anchor_batch[10],
            }
            if args.model_type != "distilbert":
                anchor_inputs["token_type_ids"] = (
                    anchor_batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )

            # 准备攻击样本的输入
            attack_inputs = {
                "input_ids": attack_batch[0],
                "attention_mask": attack_batch[1],
                "labels": attack_batch[3],
                "nodes_index_mask": attack_batch[4],
                "adj_metric": attack_batch[5],
                "node_mask": attack_batch[6],
                "sen2node": attack_batch[7],
                "sentence_mask": attack_batch[8],
                "sentence_length": attack_batch[9],
                "batch_id": attack_batch[10],
            }
            if args.model_type != "distilbert":
                attack_inputs["token_type_ids"] = (
                    attack_batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )

            # 准备正样本的输入（用于三元组损失）
            positive_inputs = {
                "input_ids": positive_batch[0],
                "attention_mask": positive_batch[1],
                "labels": positive_batch[3],
                "nodes_index_mask": positive_batch[4],
                "adj_metric": positive_batch[5],
                "node_mask": positive_batch[6],
                "sen2node": positive_batch[7],
                "sentence_mask": positive_batch[8],
                "sentence_length": positive_batch[9],
                "batch_id": positive_batch[10],
            }
            if args.model_type != "distilbert":
                positive_inputs["token_type_ids"] = (
                    positive_batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )

            # 准备负样本的输入（用于三元组损失）
            negative_inputs = {
                "input_ids": negative_batch[0],
                "attention_mask": negative_batch[1],
                "labels": negative_batch[3],
                "nodes_index_mask": negative_batch[4],
                "adj_metric": negative_batch[5],
                "node_mask": negative_batch[6],
                "sen2node": negative_batch[7],
                "sentence_mask": negative_batch[8],
                "sentence_length": negative_batch[9],
                "batch_id": negative_batch[10],
            }
            if args.model_type != "distilbert":
                negative_inputs["token_type_ids"] = (
                    negative_batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )

            # 前向传播
            # 计算锚点与攻击样本的对比损失
            anchor_outputs, anchor_rep = model(**anchor_inputs)
            attack_outputs, attack_rep = model(**attack_inputs)

            # 分类损失
            anchor_loss = anchor_outputs[0]
            attack_loss = attack_outputs[0]
            classification_loss = (anchor_loss + attack_loss) / 2

            # 第一种对比损失（与攻击样本的对比）
            contrastive_loss_1 = compute_contrastive_loss(anchor_rep, attack_rep, args.temperature)

            # 三元组对比损失（锚点、正样本、负样本）
            positive_outputs, positive_rep = model(**positive_inputs)
            negative_outputs, negative_rep = model(**negative_inputs)

            # 计算三元组损失
            triplet_loss = compute_triplet_contrastive_loss(
                anchor_rep, positive_rep, negative_rep, margin=args.triplet_margin
            )

            # 总损失
            loss = (1 - args.contrastive_weight) * classification_loss + \
                   args.contrastive_weight * (contrastive_loss_1 + triplet_loss)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(
                "loss {}".format(
                    round(tr_loss * args.gradient_accumulation_steps / (step + 1), 4)
                )
            )
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    # 如果使用 wandb，可以取消注释以下行
                    # wandb.log({"eval/loss": loss_scalar})

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # 在每个epoch结束后进行评估和保存模型
        if args.local_rank in [-1, 0] and args.save_steps > 0 and args.do_test:
            results = evaluate(args, model, tokenizer, checkpoint=str(idx), mode='dev')
            logger.info("the results is {}".format(results))
            if results["acc"] > max_acc:
                max_acc = results["acc"]
                max_acc_f1 = results["f1"]
            if results["f1"] > max_f1:
                max_f1 = results["f1"]
                max_f1_acc = results["acc"]
            if results["f1"] > best_f1:
                best_f1 = results["f1"]

                output_dir = os.path.join(
                    args.output_dir,
                    "seed-{}".format(args.seed),
                    "checkpoint-{}-{}".format(idx, best_f1),
                )  
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # 处理分布式/并行训练
                model_to_save.save_pretrained(output_dir)
                torch.save(
                    args, os.path.join(output_dir, "training_{}.bin".format(idx))
                )

                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

    # 定义评估函数
    def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode="dev"):
        eval_task_names = (args.task_name,)
        eval_outputs_dirs = (args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = load_and_cache_examples(
                args, eval_task, tokenizer, evaluate=True, mode=mode
            )

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly.
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Evaluation
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds, out_label_ids = None, None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)
                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[3],
                        "nodes_index_mask": batch[4],
                        "adj_metric": batch[5],
                        "node_mask": batch[6],
                        "sen2node": batch[7],
                        "sentence_mask": batch[8],
                        "sentence_length": batch[9],
                    }
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2]
                            if args.model_type in ["bert", "xlnet", "albert"]
                            else None
                        )
                    outputs, _ = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                    )
            probs = preds
            eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            # 如果使用 wandb，可以取消注释以下行
            # wandb.log(
            #     {
            #         "eval/acc": result["acc"],
            #         "eval/f1": result["f1"],
            #         "eval/acc_and_f1": result["acc_and_f1"],
            #     }
            # )
        return results

# 定义数据加载与缓存函数
def load_and_cache_examples(
    args, task, tokenizer, evaluate=False, mode="train", dataset_name="", rel=""
):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(dataset_name),
            str(rel),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        if mode == "train":
            examples = processor.get_train_examples(args.with_relation, args.data_dir)
        elif mode == "dev":
            examples = processor.get_dev_examples(args.with_relation, args.data_dir)
        elif mode == "test":
            examples = processor.get_test_examples(args.with_relation, args.data_dir)

        # Check if the dataset is empty
        if len(examples) == 0:
            raise ValueError(f"{mode.capitalize()} dataset is empty. Please check your data source.")

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            mask_padding_with_zero=True,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Ensure only the first process in distributed training processes the dataset
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    # Initialize lists for additional features
    all_nodes_index_mask = []
    all_adj_metric = []
    all_node_mask = []
    all_sen2node = []
    all_sen_mask = []
    all_sen_length = []

    for f in features:
        # 生成节点掩码
        nodes_mask, node_num = generate_shaped_nodes_mask(
            f.nodes_index, args.max_seq_length, args.max_nodes_num
        )
        nmask = np.zeros(args.max_nodes_num)
        nmask[:node_num] = 1
        all_node_mask.append(nmask)

        # 生成边掩码
        adj_metric = generate_shaped_edge_mask(
            f.adj_metric, node_num, args.max_nodes_num, args.with_relation
        )
        all_nodes_index_mask.append(nodes_mask)
        all_adj_metric.append(adj_metric)

        # 生成句子到节点的掩码
        sen2node_mask = np.zeros(shape=(args.max_sentences, args.max_nodes_num))

        # 生成句子掩码
        sen_mask = np.zeros(args.max_sentences)
        sen_length = min(len(f.sen2node), args.max_sentences)
        sen_mask[:sen_length] = 1
        all_sen_mask.append(sen_mask)

        # 记录句子的长度
        all_sen_length.append(sen_length)

        # 填充 sen2node_mask
        for idx_sen, sennode_list in enumerate(f.sen2node):
            if idx_sen >= args.max_sentences:
                break
            for sennode in sennode_list:
                if sennode < args.max_nodes_num:
                    sen2node_mask[idx_sen, sennode] = 1
        all_sen2node.append(sen2node_mask)

    # 将列表转换为张量
    all_nodes_index_mask = torch.tensor(all_nodes_index_mask, dtype=torch.float)
    all_node_mask = torch.tensor(all_node_mask, dtype=torch.int)
    all_adj_metric = torch.tensor(all_adj_metric, dtype=torch.float)
    all_sen2node_mask = torch.tensor(all_sen2node, dtype=torch.float)
    all_sen_mask = torch.tensor(all_sen_mask, dtype=torch.float)
    all_sen_length = torch.tensor(all_sen_length, dtype=torch.long)

    batch_id = torch.arange(len(all_labels), dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels,
        all_nodes_index_mask,
        all_adj_metric,
        all_node_mask,
        all_sen2node_mask,
        all_sen_mask,
        all_sen_length,
        batch_id,
    )
    return dataset

# 设置默认的 output_dir
if args.output_dir is None:
    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用 dataset_name 生成 output_dir 路径
    args.output_dir = os.path.join(
        os.getcwd(), "test", f"{args.dataset_name}_{current_time}"
    )

# 获取每个标签的训练索引
def get_train_idx_by_label(dataset):
    train_idx_by_label = {}
    for i in range(2):
        train_idx_by_label[i] = [
            idx for idx in range(len(dataset)) if int(dataset[idx][3]) == i
        ]
    return train_idx_by_label

# 主运行函数
def run(conf, data_dir=None):
    args.seed = conf["seed"]
    args.data_dir = data_dir

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    print(device)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args.seed)  # 修改这里



    # 准备任务
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 加载预训练模型和分词器
    if args.local_rank not in [-1, 0]:
        # 确保分布式训练中只有第一个进程下载模型和词汇表
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        task_specific_params={
            "gcn_layer": args.gcn_layer,
            "max_nodes_num": args.max_nodes_num,
            "max_sentences": args.max_sentences,
            "max_sen_replen": args.max_sen_replen,
            "attention_maxscore": args.attention_maxscore,
            "relation_num": args.with_relation,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,                 # 自动使用本地路径
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # 加载并包装对比学习数据集
    train_dataset = load_and_cache_examples(
        args,
        args.task_name,
        tokenizer,
        evaluate=False,
        mode="train",
        dataset_name=args.dataset_name,
        rel=("relation" if args.with_relation > 0 else ""),
    )
    contrastive_train_dataset = ContrastiveDataset(train_dataset, group_size=12, seed=args.seed)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # 根据 loss_type 选择模型
    if args.loss_type == "scl":
        model = RobertaForContrastiveSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
    elif args.loss_type == "mbcl":
        model = RobertaForGraphBasedSequenceClassification_MBCL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
            mb_dataloader=DataLoader(train_dataset, sampler=SequentialSampler(train_dataset)),
            train_idx_by_label=get_train_idx_by_label(train_dataset),
        )
        model.to(args.device)
    elif args.loss_type == "rfcl":
        model = RobertaForGraphBasedSequenceClassification_RFCL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
    elif args.loss_type == "contrastive":
        model = RobertaForContrastiveSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
    elif args.loss_type == 'normal':  # Baseline
        model = RobertaForGraphBasedSequenceClassification_CL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
    else:
        raise NotImplementedError

    if args.local_rank == 0:
        # 确保分布式训练中只有第一个进程下载模型和词汇表
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # 开始训练
    if args.loss_type != "mbcl":
        final_output = None  # 为 output_dir 设置默认值
        output_dir = None    # 在 run 函数中初始化 output_dir
        if args.do_train:
            global_step, tr_loss, res, output_dir = train(
                args, contrastive_train_dataset, model, tokenizer
            )
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        final_output = None
        if output_dir is not None:
            final_output = output_dir
        # 保存最佳实践：如果使用默认名称保存模型，可以使用 from_pretrained() 重新加载
        if args.do_train and (
            args.local_rank == -1 or torch.distributed.get_rank() == 0
        ):
            # 如果需要，创建输出目录
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # 使用 save_pretrained() 保存训练好的模型、配置和分词器
            # 然后可以使用 from_pretrained() 重新加载
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # 处理分布式/并行训练
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # 良好实践：将训练参数与训练好的模型一起保存
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    else:
        # 对于 'mbcl' loss_type 的处理
        if args.do_train:
            memory_dataset = load_and_cache_examples(
                args,
                args.task_name,
                tokenizer,
                evaluate=False,
                mode="train",
                dataset_name=args.dataset_name,
                rel=("relation" if args.with_relation > 0 else ""),
            )
            mb_sampler = SequentialSampler(memory_dataset)
            train_idx_by_label = get_train_idx_by_label(memory_dataset)
            mb_dataloader = DataLoader(memory_dataset, sampler=mb_sampler, batch_size=args.train_batch_size)
            model = RobertaForGraphBasedSequenceClassification_MBCL.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
                mb_dataloader=mb_dataloader,
                train_idx_by_label=train_idx_by_label,
            )
            model.to(args.device)
            global_step, tr_loss, res, output_dir = mb_train(
                args, train_dataset, model, tokenizer, mb_dataloader
            )
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        final_output = None
        if output_dir is not None:
            final_output = output_dir
        # 保存最佳实践：如果使用默认名称保存模型，可以使用 from_pretrained() 重新加载
        if args.do_train and (
            args.local_rank == -1 or torch.distributed.get_rank() == 0
        ):
            # 如果需要，创建输出目录
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # 使用 save_pretrained() 保存训练好的模型、配置和分词器
            # 然后可以使用 from_pretrained() 重新加载
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # 处理分布式/并行训练
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # 良好实践：将训练参数与训练好的模型一起保存
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # 测试
    if args.do_test and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # 减少日志输出
        best_model_file = final_output
        logger.info("Evaluate the following checkpoints: %s", best_model_file) # 加载最佳检查点
        model = AutoModelForSequenceClassification.from_pretrained(best_model_file).to(args.device)
        # model = AutoModelForSequenceClassification.from_pretrained(best_model_file, ignore_mismatched_sizes=True).to(args.device)
        results = evaluate(args, model, tokenizer, prefix='', mode='val')
        if args.do_ray:
            tune.report(test_accuracy = results["acc"],
                        test_f1 = results["f1"],
                        )
    return res

# 主函数
def main():
    data_dir = os.path.abspath("/data/Content Moderation/AI detect/Adversarial/combined/")
    if args.do_ray == True:
        import ray
        ray.init()
        config = {
            "seed": tune.choice([10, 11, 12, 13, 14, 15]),
        }  # 可以列出任何随机种子
        scheduler = ASHAScheduler(metric="accuracy", mode="max")
        reporter = CLIReporter(
            metric_columns=[
                "accuracy",
                "max_acc_f1",
                "f1",
                "max_f1_acc",
                "test_accuracy",
                "test_f1",
            ]
        )
        result = tune.run(
            partial(run, data_dir=data_dir),
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=config,
            num_samples=8,
            scheduler=scheduler,
            progress_reporter=reporter,
        )
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print(
            "Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]
            )
        )
    else:
        for seed in [10, 11, 12, 13, 14, 15]:  # 可以列出任何随机种子
            config = {
                "seed": seed,
            }
            run(config, data_dir)

if __name__ == "__main__":
    main()
