#!/usr/bin/env python3
"""Build both notebooks in notebooks/ and MANIFEST.json (standard library only).

The notebook is self-contained: it embeds the verbatim ECERC files from ecerc_original/ and the analysis
library src/l2_posthoc.py as base64, and checks every file's SHA-256 when it runs.

    python3 research/limitation2/tools/build_notebook.py
"""
import base64
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                   # research/limitation2
REPO = os.path.dirname(os.path.dirname(ROOT))  # repository root
ORIGINALS = ['model.py', 'dataloader.py', 'inference.py', 'loss.py']
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_posthoc_LA.ipynb')
LEAK_NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_feature_leak_check.ipynb')
PATHB_NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_pathB_representation.ipynb')
EXPERT_NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_expert_fusion.ipynb')
LLM_NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_llm_fusion.ipynb')
OBSERVED_DATA_SHA = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'  # run_colab_1
HF = 'https://huggingface.co/zt-ai/ECERC/resolve/main/MELD/'
CHECKPOINTS = [
    ('C1', 'ECERC_MODEL.pkl', '0f44a6e546592a3f7843342c0f72dcfa591eab1f99cdbd8a7d4ea7aed607441c'),
    ('C2', 'ECERC_MODEL_2.pkl', '47472e9ed138934737d3516be3bd630a16edad14971fe29ce5002e1d7bbdf28c'),
    ('C3', 'ECERC_MODEL_3.pkl', 'd5104e06fb050410e0720b9de4310474339cd2494eb98e2f57f9df89a609bf5c'),
]
DATA_FILE_ID = '1IQokygWgjaZuSXvCsfIcUzj1Gu69hDCb'
DATA_FILENAME = 'meld_emotion_semantic_features_roberta.pkl'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def code_files():
    files = {}
    for name in ORIGINALS:
        copy, orig = os.path.join(ROOT, 'ecerc_original', name), os.path.join(REPO, 'MELD', name)
        assert sha256(copy) == sha256(orig), f'ecerc_original/{name} differs from MELD/{name}'
        files[name] = copy
    files['l2_posthoc.py'] = os.path.join(ROOT, 'src', 'l2_posthoc.py')
    return files


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.strip('\n').splitlines(keepends=True))


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=text.strip('\n').splitlines(keepends=True))


CONFIG_CELL = '''
# ===== Configuration: leave as is. (Set TRUST_DATA_PICKLE = True only if the data cell asks you to.) =====
import os

TRUST_DATA_PICKLE = False

if os.path.isdir('/kaggle/working'):
    WORK_DIR = '/kaggle/working/ecerc_l2'
elif os.path.isdir('/content'):
    WORK_DIR = '/content/ecerc_l2'
else:
    WORK_DIR = os.path.abspath('ecerc_l2_work')
HF = '%(hf)s'
CONFIG = dict(
    work_dir=WORK_DIR,
    data_file_id='%(data_id)s',            # authors' Google Drive (linked from the official ECERC README)
    data_filename='%(data_name)s',
    checkpoints=[%(ckpts)s],
    trust_data_pickle=TRUST_DATA_PICKLE,
    expected_split_utterances=dict(train=9989, valid=1109, test=2610),
    expected_split_dialogues=dict(train=[1038, 1039], valid=114, test=280),
    min_dev_acc=0.55,                       # sanity floor; the dev majority-class rate is 42.4%%
    bootstrap_B=10000, power_B=2000, boot_seed=20251010, fold_seed=12345, n_folds=5,
    offline=False,                          # True only for the local smoke test
)
os.makedirs(WORK_DIR, exist_ok=True)
print('Working directory:', WORK_DIR)
'''


def code_files_cell(files, header, tail):
    """Cell that writes the embedded files (base64), checks their SHA-256 and runs `tail`."""
    blobs = []
    for name, path in files.items():
        with open(path, 'rb') as f:
            raw = f.read()
        b64 = base64.encodebytes(raw).decode('ascii')
        blobs.append(f"    '{name}': ('{hashlib.sha256(raw).hexdigest()}', '''\n{b64}'''),")
    return (header +
            'import base64, hashlib, importlib, sys\n\n'
            "CODE_DIR = os.path.join(CONFIG['work_dir'], 'code')\n"
            'os.makedirs(CODE_DIR, exist_ok=True)\n'
            'CODE_FILES = {  # name: (sha256, base64 content)\n' + '\n'.join(blobs) + '\n}\n'
            'for name, (digest, blob) in CODE_FILES.items():\n'
            '    data = base64.b64decode(blob)\n'
            "    assert hashlib.sha256(data).hexdigest() == digest, f'{name} is corrupted'\n"
            "    with open(os.path.join(CODE_DIR, name), 'wb') as f:\n"
            '        f.write(data)\n'
            "print('Code files written and verified:', ', '.join(CODE_FILES))\n"
            'if CODE_DIR not in sys.path:\n'
            '    sys.path.insert(0, CODE_DIR)\n' + tail)


def write_notebook(path, cells, gpu=True):
    meta = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                language_info=dict(name='python'))
    meta = (dict(accelerator='GPU', colab=dict(provenance=[], gpuType='T4'), **meta) if gpu
            else dict(colab=dict(provenance=[]), **meta))
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=meta)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')


def build():
    files = code_files()
    code_cell = code_files_cell(
        files, '# Writes the verbatim ECERC files and the analysis library, checks their SHA-256, then sets up.\n',
        'import l2_posthoc as L\nimportlib.reload(L)\nS = L.stage_setup(CONFIG, CODE_DIR)\n')
    ckpts = ''.join(f"\n        dict(name='{n}', url=HF + '{f}', sha256='{h}')," for n, f, h in CHECKPOINTS) + '\n    '
    cells = [
        md('''
# ECERC Limitation 2: post-hoc logit adjustment on the released MELD checkpoints

One run, about 10 to 20 minutes, does everything:

1. Downloads the three released ECERC MELD checkpoints (HuggingFace, SHA-256 checked) and the authors' preprocessed MELD features (Google Drive).
2. Checks on the **dev** set that this pipeline reproduces the original ECERC code exactly.
3. Chooses the logit-adjustment strength τ and the control settings on the **dev** set only, runs the dev diagnostics, and **freezes** every decision in `frozen_config.json`.
4. Only then runs the **test** set, once, with bootstrap confidence intervals and the pre-registered verdict.
5. Packs everything into `ecerc_l2_results.zip` and downloads it.

**How to run:** Runtime → Change runtime type → **T4 GPU** (CPU also works, just slower) → Save, then Runtime → **Run all**.
Don't edit cells or re-run single cells. If a cell stops with an error, copy the error text and send it back.
'''),
        code(CONFIG_CELL % dict(hf=HF, data_id=DATA_FILE_ID, data_name=DATA_FILENAME, ckpts=ckpts)),
        md('''
## 1. Code
The next cell contains the four original ECERC files (verbatim copies of `MELD/model.py`, `dataloader.py`, `inference.py`, `loss.py`) and the analysis library `l2_posthoc.py`, embedded as base64 and checked by SHA-256. You don't need to read it. Readable copies are in `research/limitation2/` of the repository.
'''),
        code(code_cell),
        md('''
## 2. Downloads (checkpoints from HuggingFace, MELD features from the authors' Google Drive)
If the Google Drive download fails (quota or permission message), follow the printed manual steps, then **Run all** again.
'''),
        code('L.stage_download(S)'),
        md('''
## 3. Data checks (no test labels are read)
Loads the data with a restricted unpickler that can only rebuild NumPy arrays. If it stops with a *blocked global* message, set `TRUST_DATA_PICKLE = True` in the first code cell and **Run all** again.
'''),
        code('L.stage_load_data(S)'),
        md('''
## 4. Fidelity checks on the dev set
For each checkpoint: strict weight loading into the unchanged ECERC class, two identical dev passes, and an exact match with the original `train_or_eval_model` from `MELD/inference.py`.
'''),
        code('L.stage_fidelity(S)'),
        md('''
## 5. Dev-only selection and diagnostics
Chooses τ (logit adjustment) and δ (Neutral-only control) on dev macro-F1, estimates the gain with cross-fitting, projects statistical power, and runs the diagnostics (separability, train vs dev, emotion shifts, branch margins, error consistency, calibration).
'''),
        code('L.stage_dev_analysis(S)'),
        md('''
## 6. Freeze
Writes `frozen_config.json` with every decision. The test evaluation reads its rules only from this file.
'''),
        code('L.stage_freeze(S)'),
        md('''
## 7. Test evaluation (runs once)
Test dialogues are run and test labels read for the first time here.
'''),
        code('L.stage_test(S)'),
        md('''
## 8. Results
Prints the summary and downloads `ecerc_l2_results.zip`. If the download doesn't start, open the Files panel (folder icon on the left), find `ecerc_l2/ecerc_l2_results.zip`, and download it from its menu.
'''),
        code('''
zpath = L.stage_package(S)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    print('Not running on Colab: download this file manually ->', zpath)
'''),
    ]
    write_notebook(NOTEBOOK, cells)
    return files


LEAK_CONFIG_CELL = '''
# ===== Configuration: leave as is. (Set TRUST_DATA_PICKLE = True only if the data cell asks you to.) =====
import os

TRUST_DATA_PICKLE = False

if os.path.isdir('/kaggle/working'):
    WORK_DIR = '/kaggle/working/ecerc_l2'
elif os.path.isdir('/content'):
    WORK_DIR = '/content/ecerc_l2'
else:
    WORK_DIR = os.path.abspath('ecerc_l2_work')
CONFIG = dict(
    work_dir=WORK_DIR,
    data_file_id='%(data_id)s',            # authors' Google Drive (linked from the official ECERC README)
    data_filename='%(data_name)s',
    expected_data_sha256='%(data_sha)s',  # observed in run_colab_1
    trust_data_pickle=TRUST_DATA_PICKLE,
    expected_utterances=dict(train=9989, valid=1109),
    n_folds=5, fold_seed=2026, C=1.0, max_iter=1000,   # the same probe as diagnostic D3 of the first run
    bootstrap_B=2000, boot_seed=777,
    offline=False,                          # True only for the local smoke test
)
os.makedirs(WORK_DIR, exist_ok=True)
print('Working directory:', WORK_DIR)
'''


def build_leak():
    files = {'l2_posthoc.py': os.path.join(ROOT, 'src', 'l2_posthoc.py'),
             'l2_feature_leak.py': os.path.join(ROOT, 'src', 'l2_feature_leak.py')}
    code_cell = code_files_cell(
        files, '# Writes the analysis code, checks its SHA-256, then sets up.\n',
        'import l2_posthoc\nimportlib.reload(l2_posthoc)\nimport l2_feature_leak as LK\nimportlib.reload(LK)\n'
        'S = LK.stage_setup(CONFIG, CODE_DIR)\n')
    cells = [
        md('''
# ECERC Limitation 2, follow-up: do the text emotion features carry the training labels?

The first experiment showed that ECERC's rare-emotion errors are not a decision-threshold problem. One lead stood out: a simple classifier on the text emotion features fits rare emotions almost perfectly on training data but fails on dev. That would happen if the RoBERTa model that produced these features, fine-tuned on the MELD training utterances, had baked the training labels into them.

This notebook tests that in about 10 minutes, using only the **train and dev** splits; the test set is never read:
1. Linear classifiers are trained on 4/5 of the training dialogues and scored on the held-out 1/5 *and* on dev. Without a leak, the two scores should be close.
2. The same is done with plain RoBERTa features (not fine-tuned) as a control, plus audio and visual features.
3. The pre-registered rule (shown in the output) gives the verdict.

**How to run:** a CPU runtime is enough (Runtime → Change runtime type → CPU), then Runtime → **Run all**. Don't edit cells. If a cell stops with an error, send the error text.
'''),
        code(LEAK_CONFIG_CELL % dict(data_id=DATA_FILE_ID, data_name=DATA_FILENAME, data_sha=OBSERVED_DATA_SHA)),
        md('''
## 1. Code
The analysis code (`l2_posthoc.py` from the first experiment and `l2_feature_leak.py`), embedded as base64 and checked by SHA-256. You don't need to read it. Readable copies are in `research/limitation2/src/`.
'''),
        code(code_cell),
        md('''
## 2. Data (the authors' MELD features from Google Drive)
If the download fails, follow the printed manual steps, then **Run all** again.
'''),
        code('LK.stage_download(S)'),
        md('''
## 3. Load the train and dev splits (the test split is never read)
If it stops with a *blocked global* message, set `TRUST_DATA_PICKLE = True` in the first code cell and **Run all** again.
'''),
        code('LK.stage_load(S)'),
        md('''
## 4. Linear classifiers and nearest neighbours
20 classifiers (4 feature types × 5 folds) run in parallel, which takes a few minutes. Then a nearest-neighbour check.
'''),
        code('LK.stage_probes(S)\nLK.stage_nearest_neighbour(S)'),
        md('''
## 5. Verdict (pre-registered rule)
'''),
        code('LK.stage_decide(S)'),
        md('''
## 6. Results
Prints the summary and downloads `ecerc_l2_leak_results.zip`. If the download doesn't start, open the Files panel (folder icon on the left), find `ecerc_l2/ecerc_l2_leak_results.zip`, and download it from its menu.
'''),
        code('''
zpath = LK.stage_package(S)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    print('Not running on Colab: download this file manually ->', zpath)
'''),
    ]
    write_notebook(LEAK_NOTEBOOK, cells, gpu=False)
    return files


PATHB_CONFIG_CELL = """
# ===== Configuration: leave as is. (Set TRUST_DATA_PICKLE = True only if the data cell asks you to.) =====
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # deterministic GPU matrix products

TRUST_DATA_PICKLE = False

if os.path.isdir('/kaggle/working'):
    WORK_DIR = '/kaggle/working/ecerc_l2'
elif os.path.isdir('/content'):
    WORK_DIR = '/content/ecerc_l2'
else:
    WORK_DIR = os.path.abspath('ecerc_l2_work')
CONFIG = dict(
    work_dir=WORK_DIR,
    data_file_id='%(data_id)s',            # authors' Google Drive (linked from the official ECERC README)
    data_filename='%(data_name)s',
    expected_data_sha256='%(data_sha)s',  # observed in run_colab_1
    trust_data_pickle=TRUST_DATA_PICKLE,
    expected_split_utterances=dict(train=9989, valid=1109, test=2610),
    expected_split_dialogues=dict(train=[1038, 1039], valid=114, test=280),
    epochs=40,                              # as in MELD/train.py
    stage1_seeds=[101, 102, 103], stage2_seeds=list(range(201, 211)),
    gate_min_gain=0.5,                      # stage 2 runs only if the dev winner beats the original by >= 0.5
    bootstrap_B=10000, boot_seed=20261011,
    offline=False,                          # True only for the local smoke test
)
os.makedirs(WORK_DIR, exist_ok=True)
print('Working directory:', WORK_DIR)
"""


def build_pathb():
    files = code_files()
    files['l2_pathb.py'] = os.path.join(ROOT, 'src', 'l2_pathb.py')
    code_cell = code_files_cell(
        files, '# Writes the verbatim ECERC files and the analysis code, checks their SHA-256, then sets up.\n',
        'import l2_posthoc\nimportlib.reload(l2_posthoc)\nimport l2_pathb as LB\nimportlib.reload(LB)\n'
        'S = LB.stage_setup(CONFIG, CODE_DIR)\n')
    cells = [
        md("""
# ECERC Limitation 2, Path B: can training-time changes improve the rare emotions?

Two experiments ruled out a decision-threshold fix and a strong feature leak. This one retrains ECERC (same architecture, same features file) to see whether changing what it learns helps the rare emotions.

**Stage 1, dev only** (3 seeds each). Each candidate is compared with the original training recipe (C0):
- **C1:** use the average of the four RoBERTa layers already in the data file, instead of only the first.
- **C2:** a contrastive loss that pulls each emotion towards its own class and pushes apart the confused pairs (Fear vs Sadness, Anger and Surprise; Disgust vs Anger).
- **C3:** an extra head that also predicts MELD's own sentiment labels during training.
- **C4:** a 3x higher learning rate (the released models under-fit their own training data).
- **C5:** the helpful ones combined.

Extra heads exist only during training; every evaluated model is the original ECERC architecture.

**Freeze:** the dev winner is fixed in `frozen_config_pathb.json`. If it doesn't beat the original by at least 0.5 dev macro-F1, the run stops there.

**Stage 2, confirmatory:** 10 new seeds × {original, winner}, with the same initialisation and data order per seed. The test set is evaluated **once**, at the end, against the pre-registered criteria.

Expect roughly 1 to 2 hours on a GPU. On Kaggle, run it with **Save Version → Save & Run All (Commit)** so it keeps going after you close the browser.
"""),
        code(PATHB_CONFIG_CELL % dict(data_id=DATA_FILE_ID, data_name=DATA_FILENAME, data_sha=OBSERVED_DATA_SHA)),
        md("""
## 1. Code
The four original ECERC files (verbatim), `l2_posthoc.py` from the first experiment, and `l2_pathb.py`, embedded as base64 and checked by SHA-256. Readable copies are in `research/limitation2/`.
"""),
        code(code_cell),
        md("""
## 2. Data (the authors' MELD features from Google Drive; needs internet)
"""),
        code('LB.stage_download(S)'),
        md("""
## 3. Data checks and datasets (only train and dev labels are read)
"""),
        code('LB.stage_prepare(S)'),
        md("""
## 4. Stage 1: dev screen (about 27 training runs)
"""),
        code('LB.stage_screen(S)'),
        md("""
## 5. Freeze
"""),
        code('LB.stage_freeze(S)'),
        md("""
## 6. Stage 2: confirmatory training (20 runs; skipped if the gate failed)
"""),
        code('LB.stage_confirm(S)'),
        md("""
## 7. Test evaluation (runs once)
"""),
        code('LB.stage_test(S)'),
        md("""
## 8. Results
Prints the summary and writes `ecerc_l2_pathb_results.zip`. On Kaggle, find it under **Output** (`ecerc_l2/ecerc_l2_pathb_results.zip`) once the version finishes.
"""),
        code("""
zpath = LB.stage_package(S)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    print('Download this file from the Output panel ->', zpath)
"""),
    ]
    write_notebook(PATHB_NOTEBOOK, cells, gpu=True)
    return files


EXPERT_CONFIG_CELL = """
# ===== Configuration: leave as is. (Set TRUST_DATA_PICKLE = True only if the data cell asks you to.) =====
import os

TRUST_DATA_PICKLE = False

if os.path.isdir('/kaggle/working'):
    WORK_DIR = '/kaggle/working/ecerc_l2'
elif os.path.isdir('/content'):
    WORK_DIR = '/content/ecerc_l2'
else:
    WORK_DIR = os.path.abspath('ecerc_l2_work')
HF = '%(hf)s'
CONFIG = dict(
    work_dir=WORK_DIR,
    data_file_id='%(data_id)s',            # authors' Google Drive (linked from the official ECERC README)
    data_filename='%(data_name)s',
    checkpoints=[%(ckpts)s],
    trust_data_pickle=TRUST_DATA_PICKLE,
    expected_split_utterances=dict(train=9989, valid=1109, test=2610),
    expected_split_dialogues=dict(train=[1038, 1039], valid=114, test=280),
    min_dev_acc=0.55,
    expert_model='FacebookAI/roberta-large',  # the same encoder family as ECERC's text features
    expert_seeds=[1, 2, 3], expert_epochs=6, expert_lr=1e-5, expert_batch=16, expert_max_len=256,
    expert_min_dev_mf1=35.0,                # one retry for a seed whose best dev macro-F1 stays below this
    gate_min_gain=0.5,                      # the test runs only if the cross-fitted dev gain is >= 0.5
    n_folds=5, fold_seed=12345, power_B=2000, bootstrap_B=10000, boot_seed=20261012,
    offline=False,                          # True only for the local smoke test
)
os.makedirs(WORK_DIR, exist_ok=True)
print('Working directory:', WORK_DIR)
"""


def build_expert():
    files = code_files()
    files['l2_expert.py'] = os.path.join(ROOT, 'src', 'l2_expert.py')
    code_cell = code_files_cell(
        files, '# Writes the verbatim ECERC files and the analysis code, checks their SHA-256, then sets up.\n',
        'import l2_posthoc\nimportlib.reload(l2_posthoc)\nimport l2_expert as LE\nimportlib.reload(LE)\n'
        'S = LE.stage_setup(CONFIG, CODE_DIR)\n')
    ckpts = ''.join(f"\n        dict(name='{n}', url=HF + '{f}', sha256='{h}')," for n, f, h in CHECKPOINTS) + '\n    '
    cells = [
        md("""
# ECERC Limitation 2, Experiment 4: a context-aware text expert fused with ECERC

Runs 1 to 3 showed that ECERC's rare-emotion ceiling comes from its precomputed input features. Its decision rule, objective and training don't move it. This experiment adds evidence ECERC never sees: a **RoBERTa-large** model fine-tuned end to end on each utterance's text together with the preceding dialogue. It uses the TelME-style prompt "... Now Speaker k feels `<mask>`" and past context only, the same task setting as ECERC.

Its predictions are combined with each released ECERC checkpoint through a single weight chosen on dev: 3 checkpoints × 3 expert seeds = 9 pairs.

The test runs **once**, and only if a cross-fitted dev estimate shows at least +0.5 macro-F1, after every decision is frozen.

Expect roughly 1 to 1.5 hours on a GPU. On Kaggle, run it with **Save Version → Save & Run All (Commit)**.
"""),
        code(EXPERT_CONFIG_CELL % dict(hf=HF, data_id=DATA_FILE_ID, data_name=DATA_FILENAME, ckpts=ckpts)),
        md("""
## 1. Code
The four original ECERC files (verbatim), `l2_posthoc.py` from the first experiment, and `l2_expert.py`, embedded as base64 and checked by SHA-256. Readable copies are in `research/limitation2/`.
"""),
        code(code_cell),
        md("""
## 2. Downloads, data checks and text inputs (only train and dev labels are read)
"""),
        code('LE.stage_prepare(S)'),
        md("""
## 3. Released ECERC checkpoints on dev
"""),
        code('LE.stage_ecerc_dev(S)'),
        md("""
## 4. Fine-tune the text expert (3 seeds, about 20 minutes each on a T4)
"""),
        code('LE.stage_train_experts(S)'),
        md("""
## 5. Dev-only fusion weight, cross-fitted forecast and gate
"""),
        code('LE.stage_dev_analysis(S)'),
        md("""
## 6. Freeze
"""),
        code('LE.stage_freeze(S)'),
        md("""
## 7. Test evaluation (runs once; skipped if the gate failed)
"""),
        code('LE.stage_test(S)'),
        md("""
## 8. Results
Prints the summary and writes `ecerc_l2_expert_results.zip`. On Kaggle it's under **Output** (`ecerc_l2/ecerc_l2_expert_results.zip`).
"""),
        code("""
zpath = LE.stage_package(S)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    print('Download this file from the Output panel ->', zpath)
"""),
    ]
    write_notebook(EXPERT_NOTEBOOK, cells, gpu=True)
    return files


LLM_CONFIG_CELL = """
# ===== Configuration: leave as is. (Set TRUST_DATA_PICKLE = True only if the data cell asks you to.) =====
import os

TRUST_DATA_PICKLE = False

if os.path.isdir('/kaggle/working'):
    WORK_DIR = '/kaggle/working/ecerc_l2'
elif os.path.isdir('/content'):
    WORK_DIR = '/content/ecerc_l2'
else:
    WORK_DIR = os.path.abspath('ecerc_l2_work')
HF = '%(hf)s'
CONFIG = dict(
    work_dir=WORK_DIR,
    data_file_id='%(data_id)s',            # authors' Google Drive (linked from the official ECERC README)
    data_filename='%(data_name)s',
    checkpoints=[%(ckpts)s],
    trust_data_pickle=TRUST_DATA_PICKLE,
    expected_split_utterances=dict(train=9989, valid=1109, test=2610),
    expected_split_dialogues=dict(train=[1038, 1039], valid=114, test=280),
    min_dev_acc=0.55,
    llm_model='Qwen/Qwen2.5-7B-Instruct',   # Apache-2.0, no access request needed
    llm_quantize=True,                      # 4-bit QLoRA so a 7B model trains on one 16 GB T4
    llm_seeds=[1, 2],                       # one per T4; they train in parallel
    llm_epochs=1, llm_lr=1e-4, llm_micro_batch=8, llm_grad_accum=2, llm_warmup=0.05,
    llm_eval_every=0.34, llm_patience=2, llm_eval_batch=16, llm_max_len=256,
    llm_max_minutes=420,                    # hard cap on training time per seed (keeps the run inside Kaggle's 12 h)
    llm_lora_r=16, llm_lora_alpha=32, llm_lora_dropout=0.05,
    llm_lora_targets=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    llm_min_dev_mf1=35.0,                   # one retry for a seed whose best dev macro-F1 stays below this
    gate_min_gain=0.5,                      # the test runs only if the cross-fitted dev gain is >= 0.5
    n_folds=5, fold_seed=12345, power_B=2000, bootstrap_B=10000, boot_seed=20261013,
    poll_seconds=30, stagger_seconds=120,   # the second worker starts 2 minutes after the first
    offline=False,                          # True only for the local smoke test
)
os.makedirs(WORK_DIR, exist_ok=True)
print('Working directory:', WORK_DIR)
"""


def build_llm():
    files = code_files()
    for n in ('l2_expert.py', 'l2_llm.py', 'l2_llm_worker.py'):
        files[n] = os.path.join(ROOT, 'src', n)
    code_cell = code_files_cell(
        files, '# Writes the verbatim ECERC files and the analysis code, checks their SHA-256, then sets up.\n',
        'import l2_posthoc\nimportlib.reload(l2_posthoc)\nimport l2_expert\nimportlib.reload(l2_expert)\n'
        'import l2_llm as LL\nimportlib.reload(LL)\nS = LL.stage_setup(CONFIG, CODE_DIR)\n')
    ckpts = ''.join(f"\n        dict(name='{n}', url=HF + '{f}', sha256='{h}')," for n, f, h in CHECKPOINTS) + '\n    '
    cells = [
        md("""
# ECERC Limitation 2, Experiment 5: a fine-tuned LLM expert fused with ECERC

Same design as Experiment 4, with a much stronger expert: **Qwen2.5-7B-Instruct**, fine-tuned with QLoRA (4-bit base, LoRA adapters, classification head) on each utterance plus the preceding dialogue. It uses past context only, ECERC's task setting. Two seeds train **in parallel, one per T4**.

Its predictions are combined with each released ECERC checkpoint through one weight chosen on dev: 3 checkpoints × 2 seeds = 6 pairs.

The test runs **once**, only if a cross-fitted dev estimate shows at least +0.5 macro-F1, after every decision is frozen. This is the third look at the MELD test set, so a multiplicity-adjusted interval is reported as well.

Expect roughly 3 to 6 hours on Kaggle **GPU T4 x2**, with Internet **On**. Run it with **Save Version → Save & Run All (Commit)**.
"""),
        code(LLM_CONFIG_CELL % dict(hf=HF, data_id=DATA_FILE_ID, data_name=DATA_FILENAME, ckpts=ckpts)),
        md("""
## 1. Code
The four original ECERC files (verbatim) and the analysis code, embedded as base64 and checked by SHA-256. Readable copies are in `research/limitation2/`.
"""),
        code(code_cell),
        md("""
## 2. Downloads (checkpoints, MELD features, the 7B model about 15 GB), data checks and inputs (train and dev labels only)
"""),
        code('LL.stage_prepare(S)'),
        md("""
## 3. Released ECERC checkpoints on dev
"""),
        code('LL.stage_ecerc_dev(S)'),
        md("""
## 4. Fine-tune the LLM expert (2 seeds in parallel; the longest step)
Progress lines appear at each dev evaluation (every third of the epoch). Training is capped at 7 hours per seed.
"""),
        code('LL.stage_train_llm(S)'),
        md("""
## 5. Dev-only fusion weight, cross-fitted forecast and gate
"""),
        code('LL.stage_dev_analysis(S)'),
        md("""
## 6. Freeze
"""),
        code('LL.stage_freeze(S)'),
        md("""
## 7. Test evaluation (runs once; skipped if the gate failed)
"""),
        code('LL.stage_test(S)'),
        md("""
## 8. Results
Prints the summary and writes `ecerc_l2_llm_results.zip`. On Kaggle it's under **Output** (`ecerc_l2/ecerc_l2_llm_results.zip`).
"""),
        code("""
zpath = LL.stage_package(S)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    print('Download this file from the Output panel ->', zpath)
"""),
    ]
    write_notebook(LLM_NOTEBOOK, cells, gpu=True)
    return files


def write_manifest(files):
    manifest = dict(
        description='Hashes and sources for research/limitation2. Regenerate with tools/build_notebook.py.',
        original_files={f'MELD/{n}': sha256(os.path.join(REPO, 'MELD', n)) for n in ORIGINALS},
        verbatim_copies={f'ecerc_original/{n}': sha256(p) for n, p in files.items() if n in ORIGINALS},
        analysis_library={f'src/{n}': sha256(os.path.join(ROOT, 'src', n))
                          for n in ('l2_posthoc.py', 'l2_feature_leak.py', 'l2_pathb.py', 'l2_expert.py', 'l2_llm.py', 'l2_llm_worker.py')},
        notebook={os.path.relpath(p, ROOT): sha256(p) for p in (NOTEBOOK, LEAK_NOTEBOOK, PATHB_NOTEBOOK, EXPERT_NOTEBOOK, LLM_NOTEBOOK)},
        checkpoints={n: dict(url=HF + f, sha256=h) for n, f, h in CHECKPOINTS},
        in_repo_checkpoint={'MELD/ECERC_MODEL.pkl': sha256(os.path.join(REPO, 'MELD', 'ECERC_MODEL.pkl'))},
        data=dict(file=DATA_FILENAME, google_drive_file_id=DATA_FILE_ID,
                  source='ECERC README -> Drive folder 1DGoPEBvMgMOge7u20wjKn6slYe4awyQi -> meld/',
                  sha256=OBSERVED_DATA_SHA, sha256_note='observed in run_colab_1; no checksum is published'))
    with open(os.path.join(ROOT, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    files = build()
    build_leak()
    build_pathb()
    build_expert()
    build_llm()
    write_manifest(files)
    print('wrote', ', '.join(os.path.relpath(p, REPO) for p in (NOTEBOOK, LEAK_NOTEBOOK, PATHB_NOTEBOOK, EXPERT_NOTEBOOK, LLM_NOTEBOOK)), 'and MANIFEST.json')
