import torch
import copy

from config import CONFIG
from trainer import train_model, calculate_fim, train_forget_multiple
from visualization import visualize
from evaluate import train_judge, evaluate_metrics
from latent_visualizer import generate_all_plots, plot_morph_digits, plot_scatter_highlighted
from entanglement_visualization import plot_scatter_separated

if __name__ == "__main__":

    original_vae = train_model()
    visualize(original_vae, "Original VAE")

    print("\n--- VISUALIZING ORIGINAL VAE ---")
    generate_all_plots(original_vae, "Original", forgot_list=[0, 1])
    plot_morph_digits(original_vae, "Original", start_digit=7, end_digit=0)
    plot_morph_digits(original_vae, "Original", start_digit=3, end_digit=9)
    plot_scatter_highlighted(original_vae, "Original")
    plot_scatter_separated(original_vae, "Original")

    fisher_matrix = calculate_fim(original_vae)

    print("\n--- STARTING EXPERIMENT A: FORGET 0 ---")
    amnesiac_vae_0 = train_forget_multiple(original_vae, fisher_matrix, labels_to_drop=[0])

    visualize(amnesiac_vae_0, "Post-Amnesia VAE 0")
    generate_all_plots(amnesiac_vae_0, "Forget_0", forgot_list=[0])
    plot_morph_digits(amnesiac_vae_0, "Forget_0", start_digit=7, end_digit=0)
    plot_morph_digits(amnesiac_vae_0, "Forget_0", start_digit=3, end_digit=9)
    plot_scatter_highlighted(amnesiac_vae_0, "Forget_0")
    plot_scatter_separated(amnesiac_vae_0, "Forget_0")
    judge = train_judge()

    evaluate_metrics(amnesiac_vae_0, judge)


    print("\n--- STARTING EXPERIMENT B: FORGET 0 & 1 ---")
    amnesiac_vae_01 = train_forget_multiple(original_vae, fisher_matrix, labels_to_drop=[0, 1])

    visualize(amnesiac_vae_01, "Post-Amnesia VAE 0,1")
    generate_all_plots(amnesiac_vae_01, "Forget_01", forgot_list=[0, 1])
    plot_morph_digits(amnesiac_vae_01, "Forget_01", start_digit=7, end_digit=0)
    plot_morph_digits(amnesiac_vae_01, "Forget_01", start_digit=3, end_digit=9)
    plot_scatter_highlighted(amnesiac_vae_01, "Forget_01")
    plot_scatter_separated(amnesiac_vae_01, "Forget_01")

    evaluate_metrics(amnesiac_vae_01, judge)