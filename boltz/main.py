import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*")

import json
import logging
import multiprocessing
import time
import os
import pickle
import platform
import tarfile
import traceback
import urllib.request
from dataclasses import asdict, dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional, List

import click
import yaml
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from rdkit import Chem
from boltz.callbacks import BoltzProgressCallback, MultiBetaProgress, PredictionConfig, STANDARD_BETAS
from boltz.console import header, step, info, success, warn, error, status, report_table
from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.mol import load_canonicals
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"

BOLTZ1_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
]

BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False

    # boltz-sample uniform scaling
    scale_uniform_beta: float = 0.0

    # laplacian graph scaling
    scale_laplacian_beta: float = 0.0
    scale_laplacian_strategy: str = ("pck",)
    scale_laplacian_index: int = (1,)
    scale_laplacian_weights: Optional[List[float]] = (None,)


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True

    # boltz-sample uniform scaling
    scale_uniform_beta: float = 0.0

    # laplacian graph scaling
    scale_laplacian_beta: float = 0.0
    scale_laplacian_strategy: str = ("pck",)
    scale_laplacian_index: int = (1,)
    scale_laplacian_weights: Optional[List[float]] = (None,)


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024
    mask_rate_msa: float = 0.1
    mask_seed_msa: int = 42


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True
    solver: str = "euler"


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    solver: str = "euler"


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = True
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    guidance_update: bool = True
    num_gd_steps: int = 20


@rank_zero_only
def download_boltz1(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        info(f"Downloading CCD dictionary to {ccd}")
        urllib.request.urlretrieve(CCD_URL, str(ccd))

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        info(f"Downloading model weights to {model}")
        for i, url in enumerate(BOLTZ1_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))
                break
            except Exception as e:
                if i == len(BOLTZ1_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


@rank_zero_only
def download_boltz2(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not mols.exists():
        info(f"Downloading and extracting CCD data to {mols}")
        urllib.request.urlretrieve(MOL_URL, str(tar_mols))
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        info(f"Downloading Boltz-2 weights to {model}")
        for i, url in enumerate(BOLTZ2_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))
                break
            except Exception as e:
                if i == len(BOLTZ2_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue

    # Download affinity model
    affinity_model = cache / "boltz2_aff.ckpt"
    if not affinity_model.exists():
        info(f"Downloading Boltz-2 affinity weights to {affinity_model}")
        for i, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(affinity_model))
                break
            except Exception as e:
                if i == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(data: Path) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        for d in data:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
    else:
        data = [data]

    return data


def filter_inputs_structure(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Filter the manifest to only include missing predictions.

    Parameters
    ----------
    manifest : Manifest
        The manifest of the input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        status(f"Found {len(existing)} existing predictions, skipping")
    elif existing and override:
        status(f"Found {len(existing)} existing predictions, will override")

    return manifest


def filter_inputs_affinity(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Check the input data and output directory for affinity.

    Parameters
    ----------
    manifest : Manifest
        The manifest.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    status("Checking input data for affinity")

    # Get all affinity targets
    existing = {
        r.id
        for r in manifest.records
        if r.affinity
        and (outdir / "predictions" / r.id / f"affinity_{r.id}.json").exists()
    }

    # Remove them from the input data
    if existing and not override:
        status(f"Found {len(existing)} existing affinity predictions, skipping")
    elif existing and override:
        status("Found existing affinity predictions, will override")

    return Manifest([r for r in manifest.records if r.id not in existing])


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def process_input(
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
) -> bool:
    """Process a single input. Returns True on success, False on failure."""
    try:
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd, mol_dir, boltz2)
        elif path.suffix in (".yml", ".yaml"):
            target = parse_yaml(path, ccd, mol_dir, boltz2)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        if to_generate:
            status(f"Generating MSA for {path.name} ({len(to_generate)} proteins)")
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)

        return True

    except Exception as e:
        # Provide detailed error message with type and traceback hint
        err_type = type(e).__name__
        err_msg = str(e) if str(e).strip() else "(no message)"
        error(f"Failed to process {path}: [{err_type}] {err_msg}")
        # Log traceback for debugging
        logging.debug(f"Traceback for {path}:\n{traceback.format_exc()}")
        return False


@rank_zero_only
def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 8192,
    use_msa_server: bool = False,
    boltz2: bool = False,
    preprocessing_threads: int = 1,
) -> Manifest:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.
    boltz2: bool, optional
        Whether to use Boltz2, by default False.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
    # Check if records exist at output path
    records_dir = out_dir / "processed" / "records"
    if records_dir.exists():
        # Load existing records
        existing = [Record.load(p) for p in records_dir.glob("*.json")]
        processed_ids = {record.id for record in existing}

        # Filter to missing only
        data = [d for d in data if d.stem not in processed_ids]

        # Nothing to do, update the manifest and return
        if data:
            status(f"Found {len(existing)} existing processed inputs, skipping")
        else:
            status("All inputs already processed")
            updated_manifest = Manifest(existing)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")

    # Create output directories
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        max_msa_seqs=max_msa_seqs,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    # Parse input data
    preprocessing_threads = min(preprocessing_threads, len(data))

    if preprocessing_threads > 1 and len(data) > 1:
        with Pool(preprocessing_threads) as pool:
            results = list(pool.imap(process_input_partial, data))
    else:
        results = [process_input_partial(path) for path in data]

    n_failed = results.count(False)
    if n_failed > 0:
        warn(f"{n_failed}/{len(data)} target(s) failed during preprocessing")

    # Load all records and write manifest
    records = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


@click.group()
def cli() -> None:
    """Boltz."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--max_parallel_samples",
    type=int,
    help="The maximum number of samples to predict in parallel. Default is None.",
    default=5,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Stop immediately if any input fails preprocessing.",
)
@click.option(
    "--dry_run",
    is_flag=True,
    help="Validate inputs and show configuration without running predictions.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to not use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz1", "boltz2"]),
    help="The model to use for prediction. Default is boltz2.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is None.",
    default=None,
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--affinity_mw_correction",
    is_flag=True,
    type=bool,
    help="Whether to add the Molecular Weight correction to the affinity value head.",
)
@click.option(
    "--sampling_steps_affinity",
    type=int,
    help="The number of sampling steps to use for affinity prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples_affinity",
    type=int,
    help="The number of diffusion samples to use for affinity prediction. Default is 5.",
    default=5,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
@click.option(
    "--mask_rate_msa",
    type=click.FloatRange(0.0, 1.0),
    default=0.1,
    help="The masking rate for MSA sequences (replace with X). Default is 0.1.",
)
@click.option(
    "--scale_uniform_beta",
    type=str,
    default=None,
    help=(
        "Beta value(s) for uniform scaling. "
        "A single value (e.g. '0.15') or comma-separated list (e.g. '-0.75,-0.30,0.30,0.75'). "
        "If not specified, runs all standard betas (-0.75 to 0.75)."
    ),
)
@click.option(
    "--scale_laplacian_beta",
    type=float,
    default=0.0,
    help=(
        "The beta value for the Laplacian scaling. "
        "Default is 0.0, which means no Laplacian scaling will be used."
    ),
)
@click.option(
    "--scale_laplacian_strategy",
    type=click.Choice(["pck", "mix"]),
    default="pck",
    help=("The Laplacian scaling strategy to use, Default is 'pck'."),
)
@click.option(
    "--scale_laplacian_index",
    type=int,
    default=1,
    help=(
        "The index of the PC to use for Laplacian scaling. "
        "Default is 1, which means the first PC will be used."
    ),
)
@click.option(
    "--scale_laplacian_weights",
    type=click.Tuple([float, float, float]),
    default=(0.5, 0.5, 0.0),
    help=("The weights to use for the Laplacian scaling mix strategy."),
)
@click.option(
    "--solver",
    type=click.Choice(["euler", "dpm_1", "dpm_2"]),
    default="euler",
    help=(
        "Diffusion solver to use. 'euler' is the default Euler-Maruyama solver. "
        "'dpm_1' is DPM-Solver-1 (first-order, similar to DDIM). "
        "'dpm_2' is DPM-Solver-2 (second-order, higher quality with fewer steps)."
    ),
)
def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 3,
    max_parallel_samples: Optional[int] = None,
    step_scale: Optional[float] = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    use_potentials: bool = False,
    model: Literal["boltz1", "boltz2"] = "boltz2",
    method: Optional[str] = None,
    affinity_mw_correction: Optional[bool] = False,
    preprocessing_threads: int = 1,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
    mask_rate_msa: float = 0.1,
    scale_uniform_beta: Optional[str] = None,
    scale_laplacian_beta: float = 0.0,
    scale_laplacian_strategy: Literal["pck", "mix"] = "pck",
    scale_laplacian_index: int = 1,
    scale_laplacian_weights: Optional[tuple[float, float, float]] = (0.5, 0.5, 0.0),
    solver: Literal["euler", "dpm_1", "dpm_2"] = "euler",
) -> None:
    """Run predictions with Boltz."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        warn("Running on CPU, this will be slow. Consider using a GPU.")

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning:
    # torch.set_float32_matmul_precision("highest")
    torch.set_float32_matmul_precision("high")

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed, verbose=False)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    input_path_display = str(data)
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Display startup banner
    header(model=model)

    # Step 1: Process inputs
    step(1, 4, "Processing inputs")
    info(f"Input: {input_path_display}")
    info(f"Output: {out_dir}")

    # Download necessary data and model
    if model == "boltz1":
        download_boltz1(cache)
    elif model == "boltz2":
        download_boltz2(cache)
    else:
        msg = f"Model {model} not supported. Supported: boltz1, boltz2."
        raise ValueError(f"Model {model} not supported.")

    # Validate inputs
    data = check_inputs(data)
    n_inputs = len(data)

    # Check method
    if method is not None:
        if model == "boltz1":
            msg = "Method conditioning is not supported for Boltz-1."
            raise ValueError(msg)
        if method.lower() not in const.method_types_ids:
            method_names = list(const.method_types_ids.keys())
            msg = f"Method {method} not supported. Supported: {method_names}"
            raise ValueError(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        boltz2=model == "boltz2",
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    n_targets = len(manifest.records)
    n_failed = n_inputs - n_targets
    if n_failed > 0:
        warn(f"{n_failed}/{n_inputs} target(s) failed during preprocessing")
        if strict:
            raise RuntimeError(f"Strict mode: {n_failed} target(s) failed preprocessing. Aborting.")
    success(f"Preprocessing complete ({n_targets} target{'s' if n_targets != 1 else ''})")

    # Filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=override,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = "fork" if platform.system() != "win32" else "spawn"
        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < devices:
            status("Devices > predictions, using minimum")
            if isinstance(devices, list):
                devices = devices[: max(1, len(filtered_manifest.records))]
            else:
                devices = max(1, min(len(filtered_manifest.records), devices))

    # Determine beta values to run
    if scale_uniform_beta is not None:
        betas = [float(x) for x in scale_uniform_beta.split(",") if x.strip()]
        multi_beta = len(betas) > 1
    else:
        betas = STANDARD_BETAS
        multi_beta = True

    # Common parameters
    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
        mask_rate_msa=mask_rate_msa,
        mask_seed_msa=seed,
    )

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": max_parallel_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }

    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = use_potentials
    steering_args.guidance_update = use_potentials

    device_str = f"cuda:{devices[0]}" if isinstance(devices, list) else f"cuda:{devices-1}" if devices > 0 else "cpu"

    if checkpoint is None:
        checkpoint = cache / ("boltz2_conf.ckpt" if model == "boltz2" else "boltz1_conf.ckpt")

    # Dry-run: show configuration and exit
    if dry_run:
        precision_str = "fp32" if model == "boltz1" else "bf16"
        step(2, 4, "Configuration (dry run)")
        info(f"Model: {model}")
        info(f"Checkpoint: {checkpoint}")
        info(f"Device: {device_str} ({precision_str})")
        info(f"Targets: {n_targets}")
        if multi_beta:
            info(f"Beta values: {len(betas)} ({betas[0]:+.2f} to {betas[-1]:+.2f})")
            info(f"Total predictions: {n_targets * len(betas) * diffusion_samples}")
        else:
            info(f"Beta: {betas[0]:+.2f}")
            info(f"Total predictions: {n_targets * diffusion_samples}")
        info(f"Output: {out_dir / 'predictions'}")
        success("Dry run complete - all inputs valid")
        return

    # Step 2: Loading model
    step(2, 4, "Loading model")
    info(f"Checkpoint: {checkpoint}")
    precision_str = "fp32" if model == "boltz1" else "bf16"
    info(f"Device: {device_str} ({precision_str})")

    # Helper: suppress Lightning's GPU/TPU log messages
    def _suppress_lightning_logs():
        """Suppress Lightning's GPU/TPU availability and checkpoint version messages."""
        logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
        logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)
        # Suppress checkpoint version mismatch warning
        warnings.filterwarnings("ignore", message=".*loaded checkpoint was produced with Lightning.*")

    # Helper: create diffusion/pairformer args for a given beta
    def make_model_args(beta: float):
        if model == "boltz2":
            diff_params = Boltz2DiffusionParams()
            diff_params.step_scale = 1.5 if step_scale is None else step_scale
            diff_params.solver = solver
            pf_args = PairformerArgsV2(
                scale_uniform_beta=beta,
                scale_laplacian_beta=scale_laplacian_beta,
                scale_laplacian_strategy=scale_laplacian_strategy,
                scale_laplacian_index=scale_laplacian_index,
                scale_laplacian_weights=scale_laplacian_weights,
            )
        else:
            diff_params = BoltzDiffusionParams()
            diff_params.step_scale = 1.638 if step_scale is None else step_scale
            diff_params.solver = solver
            pf_args = PairformerArgs(
                scale_uniform_beta=beta,
                scale_laplacian_beta=scale_laplacian_beta,
                scale_laplacian_strategy=scale_laplacian_strategy,
                scale_laplacian_index=scale_laplacian_index,
                scale_laplacian_weights=scale_laplacian_weights,
            )
        return diff_params, pf_args

    # Helper: create data module
    def make_data_module():
        if model == "boltz2":
            return Boltz2InferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
                template_dir=processed.template_dir,
                extra_mols_dir=processed.extra_mols_dir,
                override_method=method,
            )
        return BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
        )

    # Helper: load and run prediction
    def run_prediction(beta: float, output_dir: Path, callbacks: list):
        diffusion_params, pairformer_args = make_model_args(beta)

        trainer = Trainer(
            default_root_dir=out_dir,
            strategy=strategy,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=32 if model == "boltz1" else "bf16-mixed",
            logger=False,
            enable_progress_bar=False,
        )

        model_cls = Boltz2 if model == "boltz2" else Boltz1
        model_module = model_cls.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()

        trainer.predict(model_module, datamodule=make_data_module(), return_predictions=False)

    # Step 3: Run predictions
    step(3, 4, "Running predictions")
    multi_progress = None
    if not filtered_manifest.records:
        info("No new predictions to run (all targets already processed)")
    elif multi_beta:
        # Multi-beta mode
        multi_progress = MultiBetaProgress(PredictionConfig(
            model=model,
            seed=seed,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            device=device_str,
            output_dir=str(out_dir / "predictions"),
            total_predictions=len(filtered_manifest.records),
            multi_beta=True,
            betas=betas,
            input_path=input_path_display,
            output_path=str(out_dir / "predictions"),
        ))
        multi_progress.start()

        # Suppress Lightning's GPU/TPU messages to keep output clean
        _suppress_lightning_logs()

        for beta in betas:
            beta_start = time.time()
            multi_progress.update_beta(beta)

            beta_suffix = f"beta_{beta:+.2f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
            beta_out_dir = out_dir / "predictions" / beta_suffix
            beta_out_dir.mkdir(parents=True, exist_ok=True)

            pred_writer = BoltzWriter(
                data_dir=processed.targets_dir,
                output_dir=beta_out_dir,
                output_format=output_format,
                boltz2=model == "boltz2",
                progress_callback=None,
            )

            try:
                run_prediction(beta, beta_out_dir, [pred_writer])
                multi_progress.mark_beta_done(beta, time.time() - beta_start, success=True)
            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e) if str(e).strip() else "(no message)"
                warn(f"β={beta:+.2f} failed: [{err_type}] {err_msg}")
                logging.debug(f"Traceback for β={beta:+.2f}:\n{traceback.format_exc()}")
                multi_progress.mark_beta_done(beta, time.time() - beta_start, success=False)

        multi_progress.finish()
    else:
        # Single beta mode
        _suppress_lightning_logs()
        current_beta = betas[0]
        progress_config = PredictionConfig(
            model=model,
            seed=seed,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            device=device_str,
            output_dir=str(out_dir / "predictions"),
            total_predictions=len(filtered_manifest.records),
            multi_beta=False,
            current_beta=current_beta,
            input_path=input_path_display,
            output_path=str(out_dir / "predictions"),
        )
        progress_callback = BoltzProgressCallback(progress_config)

        pred_writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
            output_format=output_format,
            boltz2=model == "boltz2",
            progress_callback=progress_callback,
        )

        run_prediction(current_beta, out_dir / "predictions", [progress_callback, pred_writer])

    # Step 4: Done
    step(4, 4, "Done")
    predictions_path = out_dir / "predictions"
    if multi_beta and multi_progress is not None:
        summary = multi_progress.get_summary()
        elapsed = summary["elapsed"]
        time_str = f"{int(elapsed // 60)}m {elapsed % 60:.0f}s" if elapsed >= 60 else f"{elapsed:.1f}s"
        success(f"{summary['completed']}/{len(betas)} betas completed in {time_str}")
        if summary["failed"] > 0:
            warn(f"{summary['failed']} beta(s) failed")
    elif betas:
        success(f"Prediction complete (β={betas[0]:+.2f})")
    info(f"Output: {predictions_path}")

    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        info("Predicting property: affinity")

        # Validate inputs
        manifest_filtered = filter_inputs_affinity(
            manifest=manifest,
            outdir=out_dir,
            override=override,
        )
        if not manifest_filtered.records:
            status("Found existing affinity predictions for all inputs, skipping")
            return

        n = len(manifest_filtered.records)
        info(f"Running affinity prediction for {n} input{'s' if n > 1 else ''}")

        pred_writer = BoltzAffinityWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
        )

        data_module = Boltz2InferenceDataModule(
            manifest=manifest_filtered,
            target_dir=out_dir / "predictions",
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method="other",
            affinity=True,
        )

        predict_affinity_args = {
            "recycling_steps": 5,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # Load affinity model
        if affinity_checkpoint is None:
            affinity_checkpoint = cache / "boltz2_aff.ckpt"

        model_module = Boltz2.load_from_checkpoint(
            affinity_checkpoint,
            strict=True,
            predict_args=predict_affinity_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args={"fk_steering": False, "guidance_update": False},
            affinity_mw_correction=affinity_mw_correction,
        )
        model_module.eval()

        trainer.callbacks[1] = pred_writer
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )


def _extract_sequence(fasta: Path) -> str:
    """Extract sequence from FASTA file."""
    return "".join(
        line.strip()
        for line in fasta.open()
        if line.strip() and not line.startswith(("#", ">"))
    )


def _make_sample_yaml(msa: Path, seq: str, out_yaml: Path) -> None:
    """Create YAML recipe from FASTA sequence and MSA path."""
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": seq, "msa": str(msa.resolve())}}
        ],
    }
    with out_yaml.open("w") as f:
        yaml.dump(data, f, sort_keys=False)


@cli.command()
@click.argument("data", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--fasta", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Input FASTA file with protein sequence.")
@click.option("--msa", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Pre-computed MSA file (A3M format).")
@click.option("--out_dir", type=click.Path(path_type=Path), required=True,
              help="Output directory for predictions.")
@click.option("--scale_uniform_beta",
              default="-0.75,-0.60,-0.45,-0.30,-0.15,0.15,0.30,0.45,0.60,0.75",
              show_default=True,
              help="Comma-separated beta values for uniform scaling.")
@click.option("--diffusion_samples", type=int, default=10, show_default=True,
              help="Number of diffusion samples per beta.")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Random seed for reproducibility.")
@click.option("--num_workers", type=int, default=8, show_default=True,
              help="Number of dataloader workers.")
@click.option("--no_msa", is_flag=True, help="Skip MSA usage.")
@click.option("--max_msa_seqs", type=int, default=None,
              help="Maximum number of MSA sequences. Default is 50000.")
@click.option("--cache", type=click.Path(exists=False), default=get_cache_path,
              help="Cache directory for model files.")
@click.option("--devices", type=int, default=1, help="Number of GPU devices.")
@click.option("--accelerator", type=click.Choice(["gpu", "cpu"]), default="gpu",
              help="Accelerator type.")
@click.option("--output_format", type=click.Choice(["pdb", "mmcif"]), default="mmcif",
              help="Output structure format.")
@click.option("--solver", type=click.Choice(["euler", "dpm_1", "dpm_2"]), default="euler",
              help="Diffusion solver: euler (default), dpm_1 (DPM-Solver-1), dpm_2 (DPM-Solver-2).")
@click.option("--sampling_steps", type=int, default=200, show_default=True,
              help="Number of sampling steps. DPM solvers work well with fewer steps (e.g., 20).")
@click.option("--dry_run", is_flag=True,
              help="Validate inputs and show configuration without running predictions.")
@click.pass_context
def sample(
    ctx,
    data: Optional[Path],
    fasta: Optional[Path],
    msa: Optional[Path],
    out_dir: Path,
    scale_uniform_beta: str,
    diffusion_samples: int,
    seed: int,
    num_workers: int,
    no_msa: bool,
    max_msa_seqs: Optional[int],
    cache: str,
    devices: int,
    accelerator: str,
    output_format: str,
    solver: str,
    sampling_steps: int,
    dry_run: bool,
) -> None:
    """Run uniform scaling predictions across multiple beta values.

    Accepts either a YAML recipe file or --fasta/--msa options.

    \b
    Examples:
      boltz sample input.yaml --out_dir output
      boltz sample --fasta seq.fasta --msa seq.a3m --out_dir output
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine input YAML
    if data is not None:
        yaml_path = data
    elif fasta is not None and msa is not None:
        seq = _extract_sequence(fasta)
        yaml_path = out_dir / f"{fasta.stem}.yaml"
        if not yaml_path.exists():
            _make_sample_yaml(msa, seq, yaml_path)
            info(f"Generated YAML: {yaml_path.name}")
    else:
        raise click.UsageError("Provide either a YAML file or both --fasta and --msa options.")

    # MSA sequences setting
    if max_msa_seqs is None:
        max_msa_seqs = 0 if no_msa else 50000

    # Invoke predict once with the full beta list
    ctx.invoke(
        predict,
        data=str(yaml_path),
        out_dir=str(out_dir),
        model="boltz2",
        devices=devices,
        accelerator=accelerator,
        num_workers=num_workers,
        diffusion_samples=diffusion_samples,
        seed=seed,
        scale_uniform_beta=scale_uniform_beta,
        max_msa_seqs=max_msa_seqs,
        output_format=output_format,
        cache=cache,
        solver=solver,
        sampling_steps=sampling_steps,
        dry_run=dry_run,
    )


def _extract_ca_coords(path: Path):
    """Extract Cα coordinates and sequence from a structure file using gemmi."""
    import gemmi
    import numpy as np

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                ca = residue.find_atom("CA", "\0")
                if ca is not None:
                    coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
        break  # first model only

    return np.array(coords, dtype=np.float64)


def _parse_beta_dirname(dirname: str) -> float:
    """Parse beta value from directory name (e.g. beta_neg0p75 -> -0.75)."""
    s = dirname.replace("beta_", "")
    s = s.replace("neg", "-").replace("pos", "+")
    s = s.replace("p", ".")
    return float(s)


@cli.command()
@click.argument("predictions_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--ref", "references", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              multiple=True, required=True, help="Reference structure(s) for TM-score (PDB/CIF).")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Output TSV path. Default: PREDICTIONS_DIR/tm_scores.tsv")
def evaluate(
    predictions_dir: Path,
    references: tuple,
    output: Optional[Path],
) -> None:
    """Compute TM-scores of predictions against reference structures.

    \b
    Examples:
      boltz evaluate output/ --ref ref1.pdb --ref ref2.pdb
      boltz evaluate output/ --ref ref.pdb --output scores.tsv
    """
    import csv
    import re
    from collections import defaultdict
    from tmtools import tm_align

    header("boltz2")
    step(1, 2, "Loading structures")

    # Find prediction files
    pred_dir = predictions_dir / "predictions" if (predictions_dir / "predictions").is_dir() else predictions_dir
    pred_files = sorted(pred_dir.rglob("*_model_*.cif")) + sorted(pred_dir.rglob("*_model_*.pdb"))

    if not pred_files:
        error("No prediction files found (*_model_*.cif or *_model_*.pdb)")
        return

    info(f"Found {len(pred_files)} prediction(s)")
    info(f"References: {len(references)}")

    # Load reference coordinates
    ref_data = []
    for ref_path in references:
        ref_path = Path(ref_path)
        coords = _extract_ca_coords(ref_path)
        ref_data.append({"name": ref_path.stem, "coords": coords})
        info(f"  {ref_path.stem}: {len(coords)} residues")

    # Compute TM-scores
    step(2, 2, "Computing TM-scores")
    results = []

    for pred_file in pred_files:
        pred_coords = _extract_ca_coords(pred_file)

        # Determine beta from path (search any parent for beta_* directory)
        beta_val = 0.0
        for part in pred_file.relative_to(pred_dir).parts:
            if part.startswith("beta_"):
                beta_val = _parse_beta_dirname(part)
                break

        model_match = re.search(r"model_(\d+)", pred_file.stem)
        model_idx = int(model_match.group(1)) if model_match else 0
        target_id = pred_file.parent.name

        # Read confidence score
        conf_file = pred_file.parent / f"confidence_{target_id}_model_{model_idx}.json"
        confidence = None
        if conf_file.exists():
            try:
                with conf_file.open() as f:
                    conf_data = json.load(f)
                confidence = conf_data.get("confidence_score")
            except (json.JSONDecodeError, OSError):
                pass

        row = {"beta": beta_val, "target": target_id, "model_idx": model_idx, "confidence": confidence}

        for ref in ref_data:
            ref_coords = ref["coords"]
            if len(pred_coords) == 0 or len(ref_coords) == 0:
                row[f"tm_{ref['name']}"] = None
                row[f"rmsd_{ref['name']}"] = None
                continue

            res = tm_align(
                pred_coords,
                ref_coords,
                "A" * len(pred_coords),
                "A" * len(ref_coords),
            )
            row[f"tm_{ref['name']}"] = res.tm_norm_chain2
            row[f"rmsd_{ref['name']}"] = res.rmsd

        results.append(row)

    # Aggregate by beta
    import math

    ref_names = [r["name"] for r in ref_data]
    beta_groups = defaultdict(list)
    for r in results:
        beta_groups[r["beta"]].append(r)

    def _stats(vals):
        """Compute mean, min, max, std for a list of values."""
        if not vals:
            return None, None, None, None
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
        return mean, min(vals), max(vals), std

    # TM-score table
    tm_columns = ["Beta", "Confidence"]
    for name in ref_names:
        tm_columns.extend([f"TM ({name})", "min", "max"])

    tm_rows = []
    for beta in sorted(beta_groups.keys()):
        rows = beta_groups[beta]
        row_vals = [f"{beta:+.2f}"]

        # Confidence mean ± std
        conf_vals = [r["confidence"] for r in rows if r["confidence"] is not None]
        if conf_vals:
            c_mean, _, _, c_std = _stats(conf_vals)
            row_vals.append(f"{c_mean:.3f}\u00b1{c_std:.2f}")
        else:
            row_vals.append("N/A")

        for name in ref_names:
            tm_vals = [r[f"tm_{name}"] for r in rows if r.get(f"tm_{name}") is not None]
            mean, mn, mx, std = _stats(tm_vals)
            if mean is not None:
                row_vals.extend([f"{mean:.3f}\u00b1{std:.3f}", f"{mn:.3f}", f"{mx:.3f}"])
            else:
                row_vals.extend(["N/A", "N/A", "N/A"])

        tm_rows.append(row_vals)

    report_table("TM-score", tm_columns, tm_rows)

    # RMSD table
    rmsd_columns = ["Beta"]
    for name in ref_names:
        rmsd_columns.extend([f"RMSD ({name})", "min", "max"])

    rmsd_rows = []
    for beta in sorted(beta_groups.keys()):
        rows = beta_groups[beta]
        row_vals = [f"{beta:+.2f}"]

        for name in ref_names:
            rmsd_vals = [r[f"rmsd_{name}"] for r in rows if r.get(f"rmsd_{name}") is not None]
            mean, mn, mx, std = _stats(rmsd_vals)
            if mean is not None:
                row_vals.extend([f"{mean:.2f}\u00b1{std:.2f}", f"{mn:.2f}", f"{mx:.2f}"])
            else:
                row_vals.extend(["N/A", "N/A", "N/A"])

        rmsd_rows.append(row_vals)

    report_table("RMSD", rmsd_columns, rmsd_rows)

    # Save full detail TSV
    output_path = output or (predictions_dir / "tm_scores.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["beta", "model_idx", "confidence"]
    for name in ref_names:
        fieldnames.extend([f"tm_{name}", f"rmsd_{name}"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    success(f"Results saved to: {output_path}")


if __name__ == "__main__":
    cli()
