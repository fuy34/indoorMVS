# main code

import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

import torch

from vPlaneRecover.config import get_parser, get_cfg
from vPlaneRecover.logger import AtlasLogger
from vPlaneRecover.model import vPlaneRecNet


seed_everything(1115)

# FIXME: should not be necessary, but something is remaining
# in memory between train and val
class CudaClearCacheCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = get_cfg(args)
    model = vPlaneRecNet(cfg)

    # read train/val amount
    with open(cfg.DATASETS_TRAIN[0]) as f:
        a = f.readlines()
    with open(cfg.DATASETS_VAL[0]) as f:
        b = f.readlines()
    print('train',len(a), 'val',len(b))

    save_path = os.path.join(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION +
                             '_lr{}_bz{}_ep{}_nfrm{}_resnet{}'.format(cfg.OPTIMIZER.ADAM.LR,
                            int(cfg.DATA.BATCH_SIZE_TRAIN  * cfg.TRAINER.NUM_GPUS), cfg.TRAINER.NUM_EPOCH,
                             cfg.DATA.NUM_FRAMES_TRAIN, cfg.MODEL.RESNETS.DEPTH  ))

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    logger = AtlasLogger(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION +
                         '_lr{}_bz{}_ep{}_nfrm{}_resnet{}'.format(cfg.OPTIMIZER.ADAM.LR,
                                                                  int(cfg.DATA.BATCH_SIZE_TRAIN * cfg.TRAINER.NUM_GPUS),
                                                                  cfg.TRAINER.NUM_EPOCH,
                                                                  cfg.DATA.NUM_FRAMES_TRAIN, cfg.MODEL.RESNETS.DEPTH))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch:03d}_{step:08d}',
        save_top_k=-1,
        period=cfg.TRAINER.CHECKPOINT_PERIOD)

    # the pytorch lighting run val first and then train and finally val again (my observation)
    trainer = pl.Trainer(
        logger=logger,
        # checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint = cfg.RESUME_CKPT,
        check_val_every_n_epoch=cfg.TRAINER.CHECKPOINT_PERIOD,
        callbacks=[CudaClearCacheCallback(), checkpoint_callback, lr_monitor],
        distributed_backend='ddp',
        max_epochs=cfg.TRAINER.NUM_EPOCH,
        benchmark=True,
        gpus= cfg.TRAINER.NUM_GPUS,
        precision=cfg.TRAINER.PRECISION,
        amp_level='O1')

    trainer.fit(model)

