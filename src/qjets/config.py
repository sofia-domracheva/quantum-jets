import hashlib
import json
from dataclasses import dataclass, asdict, field

LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

@dataclass(frozen=True)
class RuntimeConfig:
    runs_dir: str = "runs"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.log_level not in LOG_LEVELS:
            raise ValueError("log_level must be one of " + ", ".join(sorted(LOG_LEVELS)))

@dataclass(frozen=True)
class RunConfig:
    seed: int = 42

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)

    def config_hash(self) -> str:
        data = asdict(self)
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:8]





    # RANDOM_STATE = 42            # fixed RNG seed for reproducibility
    # N_SAMPLES    = 1000          # working sample size N (~10^3)
    # TEST_SIZE    = 0.20          # train/test split = 80/20
    # QUBIT_DIMS   = [4, 6, 8]     # target dimensions d' = n (clipped to #features)
    # REPS         = 2             # feature-map depth r
    # ENTANGLEMENT = "linear"      # entanglement topology (linear/circular/full)
    # SVC_C        = 1.0           # SVM regularisation
    # ENCODING_RANGE = (0.0, 2.0 * np.pi)   # encoding angle range [0, 2*pi]

    # # --- label and features from the jets ---
    # TOP_K_JETS   = 3             # how many leading jets to use as features
    # RESP_LOW     = 0.9           # "well reconstructed": 0.9 <= reco/truth <= 1.1
    # RESP_HIGH    = 1.1

    # # --- real quantum computer (budget about 9 minutes) ---
    # N_HW_TRAIN   = 10            # SMALL! the kernel costs O(N^2) circuits
    # N_HW_TEST    = 6             # test points on hardware -> 6*10 = 60 circuits
    # HW_QUBITS    = 4             # n for the hardware demo (fewer = faster, less noise)
    # SHOTS_HW     = 1024          # shots per circuit on the real device
    # IBM_TOKEN    = None          # put your API key here (or save the account first)
    # IBM_INSTANCE = None          # CRN / instance name (optional)
    # HW_BACKEND_NAME = None       # pin a device, e.g. "ibm_fez"; None -> least busy

    # # Device noise. Two sources are supported:
    # #   NOISE_FROM_REAL_DEVICE = True  -> noise model built from the LIVE calibration
    # #       of a real IBM device (NoiseModel.from_backend). Calibration data are
    # #       downloaded, but the circuits are still simulated locally, so NO QPU time
    # #       is consumed. Requires a saved account.
    # #   NOISE_FROM_REAL_DEVICE = False -> stored FakeBrisbane snapshot; works offline.
    # # Note that FakeBrisbane is an Eagle r3 device (127 qubits), whereas the hardware
    # # run below uses a Heron r2 device; taking the noise model from the same device
    # # that is used for the hardware run keeps the two experiments consistent.
    # NOISE_FROM_REAL_DEVICE = True
    # NOISE_BACKEND_NAME = "ibm_marrakesh"   # None -> least busy operational device
    # SHOTS_NOISE  = 1024          # shots per circuit in the noisy simulation
    # # The noisy simulation costs O(N^2) (each pair is a separate noisy circuit), so
    # # it uses a SMALL subsample rather than the full N = 1000.
    # NOISE_N_TRAIN = 120
    # NOISE_N_TEST  = 40

