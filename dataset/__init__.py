"""create dataset and dataloader"""
import logging


def create_dataset(mode, dataset_opt, transforms=None):
    if mode == 'CardioSpike':
        from dataset.cardio_spike import CardioSpikeDataset as D
        dataset = D(**dataset_opt, transforms=transforms)
    elif mode == 'ECGHeartbeatCategorization':
        from dataset.ECGHeartbeatCategorization import ECGHeartbeatCategorization as D
        dataset = D(**dataset_opt, transforms=transforms)
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))


    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           mode))
    return dataset
