from datetime import datetime
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import mlflow

from args import get_args
from architecture import GTA
from model import Handler
from utils import get_data, SlidingWindowDataset, create_data_loader, find_epsilon, _acquire_device

# import mlflow

if __name__ == "__main__":

    # Get arguments from console
    args = get_args(train=True)

    # Get custom id for every run
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    dataset = args.dataset

    experiment = mlflow.set_experiment(experiment_name=f"{dataset}_training")
    exp_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name=id):

        # --------------------------- START TRAINING -----------------------------
        # Get data from the dataset
        (x_train, _) = get_data(dataset, mode="train", start=args.train_start, end=args.train_end)

        # Cast data into tensor objects
        # x_train = pd.DataFrame(x_train)
        n_features = x_train.shape[1]

        # We want to perform forecasting/reconstruction on all features
        # out_dim = n_features
        print(f"Proceeding with forecasting of all {n_features} input features.")

        # Construct dataset from tensor object
        train_dataset = SlidingWindowDataset(x_train, args.seq_len, args.label_len, args.pred_len, 
                                            stride=args.stride, keep_time=args.keep_time)

        print("Training:")
        # Create the data loader(s)
        train_loader, val_loader = create_data_loader(train_dataset, args.batch_size, 
                                                    args.val_split, args.shuffle_dataset) # TODO. Check that it works

        # Initialize the model
        model = GTA(
                args.num_nodes,
                args.seq_len, 
                args.label_len,
                args.pred_len, 
                args.num_levels,
                args.factor,
                args.d_model, 
                args.n_heads, 
                args.e_layers,
                args.d_layers, 
                args.d_ff,
                args.dropout, 
                args.attn,
                args.embed,
                args.dataset,
                args.activation,
                _acquire_device(args.use_gpu, args.gpu)
            )
        model = model.double()

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

        # Add a scheduler for variable learning rate
        e_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_lr) #, gamma=args.gamma_lr)

        # Set the criterion for each process: forecasting & reconstruction
        forecast_criterion = nn.MSELoss()
        # recon_criterion = nn.MSELoss()

        # Initialize the Handler module
        handler = Handler(
            model=model,
            optimizer=optimizer,
            scheduler=e_scheduler,
            keep_time=args.keep_time,
            window_size=args.seq_len,
            n_features=n_features,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            patience=args.patience,
            forecast_criterion=forecast_criterion,
            # recon_criterion=recon_criterion,
            use_cuda=args.use_gpu,
            print_every=args.print_every
            # gamma=args.gamma
        )

        # Start training
        handler.fit(train_loader, val_loader)

        # ---------------------------- END TRAINING ------------------------------

        art_uri = mlflow.get_artifact_uri()

            # Get scores for training data to be used for thresholds later on
        print("Calculating scores on training data to be used for thresholding...")
        anom_scores = handler.score(loader=train_loader, details=False)
        # Also get the ones from the validation data
        if val_loader is not None:
            val_scores = handler.score(loader=val_loader, details=False)
            anom_scores = np.concatenate((anom_scores, val_scores), axis=0)

        # get threshold using epsilon method
        if str(args.reg_level).lower() != "none":

            if args.use_mov_av:
                smoothing_window = int(args.batch_size * args.seq_len * 0.05)
                anom_scores = pd.DataFrame(anom_scores).ewm(span=smoothing_window).mean().values.flatten()

            e_thresh = find_epsilon(errors=anom_scores, reg_level=args.reg_level)
            # update_json(art_uri, "thresholds.json", {"epsilon":e_thresh})

            # Workaround to write dimensions of dataset in config
            args.__dict__['n_features'] = n_features

            mlflow.log_dict(args.__dict__, "config.txt")

            mlflow.log_dict({'anom_scores':anom_scores.tolist()}, "anom_scores.json")

            # Don't log all parameters, only some are relevant for tuning
            to_be_logged = ['window_size', 'kernel_size', 'gru_n_layers', 'gru_hid_dim', 'fc_n_layers',
                            'fc_hid_dim', 'recon_n_layers', 'recon_hid_dim', 'alpha', 'gamma', 'dropout']
            for key in to_be_logged:
                mlflow.log_param(key, args.__dict__[key])

            mlflow.pytorch.log_model(
                pytorch_model=handler.model,
                artifact_path=f"{dataset}_model",
                #registered_model_name=f"{dataset}_model",
                pip_requirements="requirements.txt"
            )

        print("Finished.")