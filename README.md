# GAN + DCGAN Reproduction (PyTorch)

   PyTorch reproduction of GAN (Goodfellow et al., 2014) on MNIST and DCGAN (Radford et al., 2016) on CelebA 64×64, with ablation studies and a Gradio demo.

   CS-320 Deep Learning · NUST · Fall 2026
   Team: Ahtisham (lead), Quratulain, Rija Faisal
   ## Papers
   - Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014 — https://arxiv.org/abs/1406.2661
   - Radford, Metz & Chintala, *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*, ICLR 2016 — https://arxiv.org/abs/1511.06434

   ## Team and roles
   | Member | Role |
   |---|---|
   | Ahtisham | Paper lead: G + D, own adversarial loss, MNIST reproduction |
   | Quratulain | Data lead: datasets, DataLoader, evaluation (FID) |
   | Rija Faisal | Experiments lead: training loop, ablations, Gradio demo, report |

   ## Data
   - MNIST: torchvision, CC BY-SA 3.0.
   - CelebA (Liu et al., ICCV 2015): non-commercial research use only. **Not stored in this repo.** We load it from a private Google Drive copy of the Kaggle mirror `jessicali9530/celeba-dataset`.

   ## How to run
   **Step 1: Get the data into your Google Drive (once).**
   Open `download_data.ipynb` in Colab  and run all cells. You need your own Kaggle API token . It saves:
   - `MyDrive/gan-dcgan-reproduction-data/data/celeba-dataset.zip` (CelebA, ~1.3 GB)
   - `MyDrive/gan-dcgan-reproduction-data/data/mnist/` (4 MNIST `.gz` files)

   *Team members:* skip Step 1. Add the shared folder `gan-dcgan-reproduction-data` to your Drive as a shortcut (Shared with me → right-click → Organize → Add shortcut → My Drive).

   **Step 2: Load the data.**
   Open `ahtisham/00_data_loading.ipynb` in Colab, set Runtime → Change runtime type → **T4 GPU**, and run all cells. It reads both datasets from your Drive and shows one batch of each.
