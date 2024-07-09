import numpy as np
import wandb
import matplotlib.pyplot as plt
import io

def log_samples_and_targets(generated_samples, targets, step):
    """
    Logs the generated samples and targets as a table of line series plots in Weights and Biases (wandb).
    
    Args:
        generated_samples (list): List of generated samples, where each sample has shape (10, 1080).
        targets (list): List of target samples, where each target has shape (1080,).
        step (int): The step number associated with the logged table.
    """
    # Create a table to store the plots
    table = wandb.Table(columns=["Sample Index", "Generated Samples", "Target"])

    # Iterate over the generated samples and targets
    for i in range(len(generated_samples)):
        # Get the current generated samples and target
        generated_samples_batch = generated_samples[i].numpy().squeeze()
        target = targets[i].numpy().squeeze()
        
        # Create a line series plot for the generated samples
        plt.figure()
        for j in range(generated_samples_batch.shape[0]):
            plt.plot(generated_samples_batch[j], label=f"Generated Sample {j+1}")
        plt.title(f"Generated Samples for Target {i+1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        
        # Convert the plot to an image array
        generated_samples_plot = io.BytesIO()
        plt.savefig(generated_samples_plot, format='png')
        generated_samples_plot.seek(0)
        generated_samples_image = plt.imread(generated_samples_plot)
        
        # Create a line series plot for the target
        plt.figure()
        plt.plot(target)
        plt.title(f"Target {i+1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        
        # Convert the plot to an image array
        target_plot = io.BytesIO()
        plt.savefig(target_plot, format='png')
        target_plot.seek(0)
        target_image = plt.imread(target_plot)
        
        # Add a row to the table with the sample index and the plot images
        table.add_data(i+1, wandb.Image(generated_samples_image), wandb.Image(target_image))
        
        plt.close('all')

    # Log the table to wandb with the specified step
    wandb.log({"Generated Samples and Targets": table}, step=step)