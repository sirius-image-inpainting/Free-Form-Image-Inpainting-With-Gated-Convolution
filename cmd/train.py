"""
Training utility of SN-PatchGAN.

Author: Max Pershin
Email: mepershin@gmail.com
Date: 2021-03-22
"""

# ==================== [IMPORT] ====================

import torch
import pytorch_lightning as pl

import model.gan as gan
import model.data as loader

# ===================== [CODE] =====================


def train():
    model = gan.SNPatchGAN()
    data = loader.PlacesDataModule()

    trainer = pl.Trainer(
            logger=True,
            max_epochs=10,
        )

    trainer.fit(model, data)



if __name__ == '__main__':
    train()

