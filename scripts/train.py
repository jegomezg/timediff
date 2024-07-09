import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import wandb

from . import loggin

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=2,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    # Define the warm-up parameters
    warm_up_epochs = 50  # Number of epochs for warm-up
    warm_up_lr_init = 1e-8  # Initial learning rate during warm-up

    # Calculate the learning rate increment per epoch during warm-up
    lr_increment = (config["lr"] - warm_up_lr_init) / warm_up_epochs

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()

        # Adjust the learning rate during warm-up period
        if epoch_no < warm_up_epochs:
            lr = warm_up_lr_init + epoch_no * lr_increment
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                wandb.log({"lr": lr}, step=epoch_no * len(train_loader) + batch_no)
                wandb.log({"train_loss": loss.item()}, step=epoch_no * len(train_loader) + batch_no)
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

                # Remove the train_batch from GPU memory
                del train_batch
                torch.cuda.empty_cache()

                if batch_no >= config["itr_per_epoch"]:
                    break

            # Apply the learning rate scheduler after the warm-up period
            if epoch_no >= warm_up_epochs:
                lr_scheduler.step()
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            # Clean up the garbage in the GPU before evaluating
            torch.cuda.empty_cache()

            print(f"Evaluating at epoch {epoch_no+1}")
            rmse, mae,all_generated_samples,all_target = evaluate(model, valid_loader, nsample=10, foldername=foldername)
            wandb.log({"val_rmse": rmse}, step=epoch_no * len(train_loader) + batch_no)
            wandb.log({"train_mae": rmse}, step=epoch_no * len(train_loader) + batch_no)
            loggin.log_samples_and_targets(generated_samples=all_generated_samples,targets=all_target,step=epoch_no * len(train_loader) + batch_no)
            
            print(f"Validation RMSE: {rmse}, MAE: {mae}")
            if rmse < best_valid_loss:
                best_valid_loss = rmse
                print(f"Best validation CRPS updated to {rmse} at epoch {epoch_no+1}")
                if foldername != "":
                    torch.save(model.state_dict(), output_path)

            # Clean up the garbage in the GPU after evaluating
            torch.cuda.empty_cache()

    if foldername != "":
        torch.save(model.state_dict(), output_path)



def evaluate(model, test_loader, nsample=10, foldername=""):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        total_samples = 0

        all_target = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target = output
                samples = samples.permute(0, 1, 3, 2).cpu()  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1).cpu()  # (B,L,K)

                samples_median = samples.median(dim=1).values
                all_generated_samples.append(samples)
                all_target.append(c_target)

                mse_current = ((samples_median - c_target) ** 2).mean(dim=[1, 2])
                mae_current = torch.abs(samples_median - c_target).mean(dim=[1, 2])

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                total_samples += samples_median.size(0)

                it.set_postfix(
                    ordered_dict={
                        "rmse": (mse_total / total_samples) ** 0.5,
                        "mae": mae_total / total_samples,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb") as f:
                all_target = torch.cat(all_target, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)
                pickle.dump([all_generated_samples, all_target], f)

            # Log the arrays directly



            rmse = (mse_total / total_samples) ** 0.5
            mae = mae_total / total_samples

            with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump([rmse, mae], f)

            print("RMSE:", rmse)
            print("MAE:", mae)

            return rmse, mae, all_generated_samples,all_target
                
                
