"ANM-OpenFold3: pair representation ↔ contact map ↔ ANM mode-driven dynamics."

# ANM
from .anm import build_hessian, anm_modes, collectivity, combo_collectivity, batch_combo_collectivity, anm_bfactors, displace

# Contact head
from .contact_head import ContactProjectionHead

# Converter
from .converter import PairContactConverter

# Coords to contact
from .coords_to_contact import coords_to_contact

# Data
from .data import ProteinContactDataset, ShardedPairReprDataset, extract_ca_coords

# Ground truth
from .ground_truth import compute_gt_probability_matrix

# Inverse
from .inverse import PairReprFromCoords

# Kirchhoff / GNM
from .kirchhoff import soft_kirchhoff, gnm_decompose

# Losses
from .losses import focal_loss, contact_loss, gnm_loss, reconstruction_loss, total_loss

# Mode combinator
from .mode_combinator import ModeCombo, collectivity_combinations, grid_combinations, random_combinations, targeted_combinations

# Mode drive
from .mode_drive import ModeDriveConfig, ModeDrivePipeline, ModeDriveResult, StepResult, compute_rmsd, kabsch_superimpose, tm_score, make_pseudo_diffusion

# OF3 diffusion (lazy import — requires openfold3-repo)
def load_of3_diffusion(*args, **kwargs):
    from .of3_diffusion import load_of3_diffusion as _load
    return _load(*args, **kwargs)

# Model
from .model import GNMContactLearner

# Train
from .train import TrainConfig, train_one_epoch, validate, train
