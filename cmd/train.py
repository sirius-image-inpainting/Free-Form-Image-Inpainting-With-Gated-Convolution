"""
Training utility of SN-PatchGAN.

Author: Max Pershin
Email: mepershin@gmail.com
Date: 2021-03-22
"""

# ==================== [IMPORT] ====================

import os
import torch
import pytorch_lightning as pl

import model.gan as gan
import model.data as loader

# ===================== [CODE] =====================


def train():
    pl.seed_everything(42)

    model = gan.SNPatchGAN()
    data = loader.PlacesDataModule()

    neptune_logger = pl.loggers.NeptuneLogger(
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            project_name="silentz/Sirius",
            experiment_name='NeptuneGAN',
            params=dict(),
        )

    trainer = pl.Trainer(
            gpus=1,                # use gpu,
            logger=neptune_logger, # neptune logger
            max_epochs=10,         # epoch count
            val_check_interval=50, # check each 50 batch
            #  track_grad_norm=2,
        )

    trainer.fit(model, data)



if __name__ == '__main__':
    train()

