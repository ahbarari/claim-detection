import numpy as np
import matplotlib.pyplot as plt
import torch

# Load the file
file_path = "thesis-codebase/data/results/deberta-base_clef,nlp4if_42_1e-3_16_adapter_fusion/cw_diabetes/attention_scores.pt"
attention_scores = torch.load(file_path)

# Initialize variables for averaging across samples
all_layer_mean_scores = None
num_samples = len(attention_scores)

# Iterate over all samples
for sample_idx in range(num_samples):
    attention_sample = attention_scores[sample_idx]

    # Process each adapter fusion in the sample
    for fusion_key, layers in attention_sample.items():
        adapter_names = fusion_key.split(",")  # Parse adapter names dynamically
        num_adapters = len(adapter_names)
        
        # Initialize a list to store mean scores for this sample
        sample_layer_mean_scores = []
        layer_indices = []  # Track layer indices

        # Process each layer
        for layer_idx, layer_data in layers.items():
            attention_matrix = np.array(layer_data["output_adapter"][0])  # Extract attention matrix
            layer_mean = np.mean(attention_matrix, axis=0)  # Compute mean for each adapter
            sample_layer_mean_scores.append(layer_mean)
            layer_indices.append(layer_idx)
        
        # Convert to numpy array for aggregation
        sample_layer_mean_scores = np.array(sample_layer_mean_scores)
        
        # Initialize aggregation array if not done yet
        if all_layer_mean_scores is None:
            all_layer_mean_scores = np.zeros_like(sample_layer_mean_scores)
        
        # Add sample's mean scores to the aggregation
        all_layer_mean_scores += sample_layer_mean_scores

# Average across all samples
all_layer_mean_scores /= num_samples

# Calculate the mean score for each adapter across all layers
adapter_means = np.mean(all_layer_mean_scores, axis=0)

# Update adapter names to include their mean scores
adapter_labels_with_means = [
    f"{name}" for name in adapter_names
]

# Plotting the heatmap for averaged attention scores
plt.figure(figsize=(8, len(layer_indices) * 0.8))
im = plt.imshow(all_layer_mean_scores, cmap="viridis", aspect="auto")

# Configure the plot
plt.colorbar(label="Mean Attention Score")
plt.xticks(
    ticks=np.arange(num_adapters),
    labels=adapter_labels_with_means,
    rotation=45,
    ha="right"
)
plt.yticks(
    ticks=np.arange(len(layer_indices)),
    labels=[f"Layer {idx}" for idx in layer_indices]
)
plt.title("Averaged Mean Attention Scores Across All Samples")
plt.xlabel("Adapters")
plt.ylabel("Layers")
plt.tight_layout()

# Add text annotation for each cell in the heatmap
for i in range(all_layer_mean_scores.shape[0]):
    for j in range(all_layer_mean_scores.shape[1]):
        value = all_layer_mean_scores[i, j]
        # Choose text color to ensure readability against background
        color = 'white' if value < (all_layer_mean_scores.max() * 0.6) else 'black'
        plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color, fontsize=8)

plt.show()

# Save the plot
plt.savefig(f"{fusion_key}_attention_heatmap.png")
