import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from experiment import CovidCardioSpikeExperiment
from utils.fix_random import fix_random_state
from utils.pretrained_utils import load_model


@hydra.main(config_path="configs/CardioSpike.yaml")
def main(cfg):
    fix_random_state(42)

    logger = TensorBoardLogger("logs")
    checkpoint_callback = ModelCheckpoint(
        filename='model_last_{epoch}_{f1_score}',
        verbose=True,
        monitor='f1_score',
        mode='max'
    )

    model = CovidCardioSpikeExperiment(cfg)
    if cfg.pretrained_checkpoint_path:
        load_model(model, cfg.pretrained_checkpoint_path)
        if hasattr(model.net, 'freeze_pretrained_layers'):
            print('FREEZING LAYER')
            model.net.freeze_pretrained_layers()
        else:
            print("LAYERS NOT BE FREEZED, add freeze_pretrained_layers to your model")


    trainer = Trainer(gpus=cfg.gpu_ids, max_epochs=cfg.train.epoches, logger=logger, limit_val_batches=cfg.train.val_steps_limit, limit_train_batches=cfg.train.train_steps_limit,
                      log_every_n_steps=cfg.train.log_freq, flush_logs_every_n_steps=cfg.train.log_freq, resume_from_checkpoint=cfg.checkpoint_path, check_val_every_n_epoch=cfg.train.val_freq,
                      precision=cfg.train.precision, deterministic=True, gradient_clip_val=cfg.train.gradient_clip_val, callbacks=[LearningRateMonitor('epoch'), checkpoint_callback])

    trainer.fit(model)


if __name__ == '__main__':
    main()
