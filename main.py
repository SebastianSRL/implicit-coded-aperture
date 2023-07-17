#!/usr/bin/env python
import os
import torch.optim
import torch.utils.data
import numpy as np 

from itertools import product
from pathlib   import Path
from core      import M2S, Dataset, load_yml, write_yml


# config file path
CONFIG_FILE = 'config.yml'
# activate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(f"Torch version: {torch.__version__}")



def main():

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    args = load_yml(CONFIG_FILE).get('config')

    if args.get('network') == 'dssp':
        stride = 1
    else:
        stride = 2

    args.get('network').get('optical_info').update({'stride': stride})
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
    # path configurations
    # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

    # Params for the experiment
    model = ['dssp']
    # use_inr = [True, False]
    # model = ['unet',]
    use_inr = [True,]
    params = product(model, use_inr)
    
    
    for i, (model, use_inr) in enumerate(params): 
        if i > 0:
            continue
    # for _ in range(1):
        # Hyperparameter tuning
        args.get("network").update({'model': model})
        args.get("network").get("optical_info").update({"use_inr": use_inr})
        
        # # Paths and conditions
        result_path = args.get("result-path")
        save_path = args.get("save-path")
        # model = args.get("network").get("model")
        # use_inr = args.get("network").get("optical_info").get("use_inr")
        inr = args.get("network").get("optical_info").get("inr_info").get("model")
        trainable = args.get("network").get("optical_info").get("trainable_mask")
        save_path = f'{result_path}/{save_path}_{model}_no_noise'
        
        if trainable:
            if use_inr:
               save_path = save_path + f'_trainable_mask_{inr}'
            else: 
                save_path = save_path + f'_trainable_mask_baseline'
        else:
            save_path = save_path + '_random_mask'
            
        # save_path = save_path + \
        #     f'_trainable_mask_{inr}' if trainable else save_path + '_random_mask'
        # save_path += f'_w_{w}_s_{s}'

        print(f'Experiment will be saved in {save_path}')

        checkpoint_path = f'{save_path}/checkpoints'
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # Save the experiment config in a .txt file
        # save_config(save_path, os.path.basename(__file__), args)

        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
        # load dataset
        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
        
        write_yml(f"{save_path}/config.yml", config=args)
        dataset = Dataset(args.get("dataset-path"), args.get("batch-size"),
                        args.get("patch-size"), args.get("workers"))
        train_loader, val_loader = dataset.get_arad_dataset(
            gen_dataset_path=None)

        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
        # load model and hyperparams
        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
        m2s_model = M2S(save_path=save_path, device=device, **args.get("network"))

        # summary model
        num_parameters = sum([l.nelement()
                            for l in m2s_model.computational_decoder.parameters()])
        print(f'Number of parameters: {num_parameters}')

        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
        # load checkpoint
        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

        if args.get("load_path"):
            m2s_model.load_checkpoint(args.get("load_path"), epoch=args.init_epoch)
            print('¡Model checkpoint loaded correctly!')

        else:
            pass
            # raise ValueError('No checkpoint path provided')

        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
        # train
        # =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

        m2s_model.train(train_loader, args.get("init-epoch"),
                        args.get("max-epoch"), val_loader=val_loader[0])


if __name__ == '__main__':
    main()
