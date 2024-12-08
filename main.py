from src.Dataset import Dataset
from src.Utils import model_chooser, optimizer_chooser, config_checker, loss_function_chooser
from src.Process import train, test
import torch
import torch.nn as nn
import yaml


if __name__ == "__main__":
    # Load config settings
    with open('./src/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    config_checker(config)

    # Load datasets
    train_dataset = Dataset(dataset_root_dir=config['datasetRootDir'],
                            batch_size=config['batchSize'],
                            mode='train',
                            point_num=config['fullPointNum'],
                            channel_num=config['channelNum'],
                            size=config['trainDataNum'])
    test_dataset = Dataset(dataset_root_dir=config['datasetRootDir'],
                           batch_size=config['batchSize'],
                           mode='test',
                           point_num=config['fullPointNum'],
                           channel_num=config['channelNum'],
                           size=config['testDataNum'])

    # Generate device, model and optimizer
    device = torch.device(config['device'])

    if config['mode'] == 'train':
        model = model_chooser(config['model'], device, config['channelNum'], config['outputNum'], k=config['k'])
        if config['checkpoint'] != 'None':
            model.load_state_dict(torch.load(config['checkpoint']))
        optimizer = optimizer_chooser(model, config['optimizer'], config['learningRate'], config['weightDecay'])
        criterion = loss_function_chooser(config['lossFunction'],
                                          device,
                                          torch.tensor([config['otherWeight'], config['holeWeight']]),
                                          config['alpha'],
                                          config['gamma'])

        if config['downSample'] == 1:
            train_data = train_dataset.generate_downsample(down_sample_num=config['downSamplePointNum'])
            test_data = None
            if config['val'] == 1:
                test_data = test_dataset.generate_downsample(down_sample_num=config['downSamplePointNum'])
            # model = model_chooser(config['model'], device, config['channelNum'], config['outputNum'])

            # optimizer = optimizer_chooser(model, config['optimizer'], config['learningRate'], config['weightDecay'])

            train(dataset=train_data,
                  model=model,
                  optimizer=optimizer,
                  seed=config['seed'],
                  epochs=config['epochs'],
                  weight_saving_path=config['weightSavingPath'] + '/' + config['model'],
                  device=device,
                  val=config['val'],
                  test_dataset=test_data,
                  tensorboard=config['enableTensorboard'],
                  tensorboard_log=config['logPath'])

        elif config['downSample'] == 0:
            train_data = train_dataset.generate_full_size()
            test_data = None
            if config['val'] == 1:
                test_data = test_dataset.generate_full_size()
            train(dataset=train_data,
                  model=model,
                  optimizer=optimizer,
                  seed=config['seed'],
                  epochs=config['epochs'],
                  weight_saving_path=config['weightSavingPath'] + '/' + config['model'],
                  device=device,
                  val=config['val'],
                  test_dataset=test_data,
                  tensorboard=config['enableTensorboard'],
                  tensorboard_log=config['logPath'],
                  sampled=config['downSamplePointNum'])

    elif config['mode'] == 'test':
        model = model_chooser(config['model'], device, config['channelNum'], config['outputNum'], k=config['k'])
        model.load_state_dict(torch.load(config['weightPath']))
        if config['downSample'] == 1:
            test_data = test_dataset.generate_downsample(down_sample_num=config['downSamplePointNum'])
            test(dataset=test_data,
                 model=model,
                 device=device)
        elif config['downSample'] == 0:
            test_data = test_dataset.generate_full_size()
            test(dataset=test_data,
                 model=model,
                 device=device,
                 sampled=config['downSamplePointNum'])
