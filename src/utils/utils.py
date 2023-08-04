import argparse
import datetime
import shutil
import sys
import os
import random
from logging import getLogger
from collections import OrderedDict
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
import wandb
from src.modeling.modeling_base_model import BetaConfig

from .logger import setup_logger


def get_main_dir():
    # 如果是使用pyinstaller打包后的执行文件，则定位到执行文件所在目录
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    # 其他情况则定位至项目根目录
    return os.path.join(os.path.dirname(__file__), '..', '..')


def get_abs_path(*name):
    fn = os.path.join(*name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(get_main_dir(), fn))


def seed_everything(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


def get_config(config_path="", opts=[]):
    base_config = OmegaConf.load(get_abs_path('configs', 'base.yml'))
    model_config = OmegaConf.load(get_abs_path('configs', config_path)) if len(config_path) > 0 else OmegaConf.create(
        "")
    cli_config = OmegaConf.from_dotlist(opts)
    config = OmegaConf.merge(base_config, model_config, cli_config)
    return config


def args_parse(config_file=''):
    parser = argparse.ArgumentParser(description="fast-bbdl")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--opts", help="Modify config options using the command-line key value", default=[],
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    config_file = args.config_file or config_file
    cfg = get_config(config_file, args.opts)
    if len(cfg.OUTPUT_DIR) == 0:
        cfg.OUTPUT_DIR = get_abs_path('outputs', cfg.MODEL.NAME, cfg.MODEL.TASK, str(datetime.date.today()))

    if not os.path.exists(cfg.OUTPUT_DIR):
        try:
            os.makedirs(cfg.OUTPUT_DIR)
        except Exception as e:
            print(e)

    name = cfg.MODEL.NAME

    logger = setup_logger(name)
    logger.info("Running with config:\n{}".format(OmegaConf.to_yaml(cfg)))
    if cfg.MODEL.SEED >= 0:
        seed_everything(cfg.MODEL.SEED)
    return cfg


def save_model(cfg, model, optimizer, lr_scheduler, global_step, epoch, save_every_step=10000, mode='step'):
    model.eval()
    if mode == 'step':
        save_dir = get_abs_path(cfg.OUTPUT_DIR, 'checkpoints', f'step{global_step + 1}')
    else:
        save_dir = get_abs_path(cfg.OUTPUT_DIR, 'checkpoints', f'epoch{epoch + 1}')
    model_to_save = model.module
    if hasattr(model_to_save, 'module'):
        model_to_save = model_to_save.module
    model_to_save.save_pretrained(save_dir)
    torch.save(optimizer.state_dict(), get_abs_path(save_dir, 'optimizer.pt'))
    torch.save(lr_scheduler.state_dict(), get_abs_path(save_dir, 'lr_scheduler.pt'))
    OmegaConf.save(cfg, get_abs_path(save_dir, 'train_config.yml'))
    model.train()
    if mode != 'step':
        old_dir = get_abs_path(cfg.OUTPUT_DIR, 'checkpoints', f'epoch{epoch}')
    else:
        old_dir = get_abs_path(cfg.OUTPUT_DIR, 'checkpoints', f'step{global_step - save_every_step + 1}')
    shutil.rmtree(old_dir, ignore_errors=True)


def init_wandb(cfg):
    if len(cfg.WANDB.KEY) > 0:
        wandb.login(key=cfg.WANDB.KEY)
    wandb.init(
        project='_'.join(cfg.OUTPUT_DIR.split('/')[-3:-1]),
        name=str(datetime.date.today()),
        # save_dir='/tmp',
        settings=wandb.Settings(start_method="fork"),
        config={'cfg': cfg})


def ema(student, teacher, decay, skip_embedding=False):
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(decay).add_((1 - decay) * param_q.detach().data)
        if skip_embedding:
            for param_q, param_k in zip(student.bert.embeddings.parameters(), teacher.bert.embeddings.parameters()):
                param_k.data = param_q.detach().data
                param_k.requires_grad = False
    teacher.eval()


def ema_scheduler(start_decay=0.999, end_decay=0.9999, schedule_step=30000, current_step=0):
    assert start_decay <= 1
    assert end_decay <= 1
    assert current_step >= 0
    gap = end_decay - start_decay
    if current_step >= schedule_step:
        return end_decay
    return start_decay + gap * (current_step / schedule_step)


def init_steps(cfg, dataloader):
    assert cfg.SOLVER.MAX_EPOCHS > 0 or cfg.SOLVER.MAX_STEPS > 0, "epoch数和step数至少需指定其中一个"
    batch_times = cfg.SOLVER.AGB * int(os.environ.get('nodes', 1)) * int(os.environ.get('gpus', 1))

    if cfg.SOLVER.MAX_STEPS <= 0:
        cfg.SOLVER.MAX_STEPS = cfg.SOLVER.MAX_EPOCHS * len(dataloader) // batch_times + 1
        if len(dataloader) % batch_times != 0:
            cfg.SOLVER.MAX_STEPS += cfg.SOLVER.MAX_EPOCHS
    if cfg.SOLVER.MAX_EPOCHS <= 0:
        cfg.SOLVER.MAX_EPOCHS = cfg.SOLVER.MAX_STEPS * batch_times // len(dataloader) + 1
    logger = getLogger(cfg.MODEL.NAME)
    logger.info(f'total train steps: {cfg.SOLVER.MAX_STEPS}, train epochs: {cfg.SOLVER.MAX_EPOCHS}')
