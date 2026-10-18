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
AUDIO_NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_audio_evidence.ipynb')
ZEROSHOT_NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_zeroshot_llm_fusion.ipynb')
OBSERVED_DATA_SHA = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'  # run_colab_1
HF = 'https://huggingface.co/zt-ai/ECERC/resolve/main/MELD/'
CHECKPOINTS = [
    ('C1', 'ECERC_MODEL.pkl', '0f44a6e546592a3f7843342c0f72dcfa591eab1f99cdbd8a7d4ea7aed607441c'),
    ('C2', 'ECERC_MODEL_2.pkl', '47472e9ed138934737d3516be3bd630a16edad14971fe29ce5002e1d7bbdf28c'),
    ('C3', 'ECERC_MODEL_3.pkl', 'd5104e06fb050410e0720b9de4310474339cd2494eb98e2f57f9df89a609bf5c'),
]
DATA_FILE_ID = '1IQokygWgjaZuSXvCsfIcUzj1Gu69hDCb'
DATA_FILENAME = 'meld_emotion_semantic_features_roberta.pkl'
# Experiment 6: 16 kHz FLAC audio extracted from the MELD videos, and frozen encoders, all pinned to a revision.
AUDIO_DATASET = dict(
    repo='ajyy/MELD_audio', revision='90175e5207b0f1fb6f3742b3d9036b0fda509086', license='gpl-3.0',
    sha256={'train.csv': '4e3907e95e7ad4fd413a5a58d807dee79ebdb0c16b050cd33c3a3ac35ddb9ebf',
            'dev.csv': '7e306369c2d1625b063132310aa30bc6979fcf0f51c783e903edfcaa1c98b15c',
            'test.csv': '2d3fdb6586bdcdf172a7197fae9244116b1103efd951db2317e6a67c7042d28e',
            'archive/train.tar.gz': 'ddf650657ed9232b27bd4789adcfc504156df3f87d4b6073c37de637b9f97aa9',
            'archive/dev.tar.gz': '1b04cc9368d66a24442666eaa32a8c2da19332fd769c9b1aa02258a605748678',
            'archive/test.tar.gz': 'aef4ba7ea47a9c2b7653db869fcbc677cb50e795f46ec1e00077a3214de1ee66'})
AUDIO_ENCODERS = [
    dict(name='whisper-large-v3', repo='openai/whisper-large-v3', revision='06f233fe06e710322aca913c1bc4249a0d71fce1',
         kind='whisper', batch=8),
    dict(name='wavlm-large', repo='microsoft/wavlm-large', revision='c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
         kind='wavlm', batch=1),
]
TEXT_ENCODER = dict(name='bge-large-en-v1.5', repo='BAAI/bge-large-en-v1.5',
                    revision='d4aa6901d3a41ba39fb536a557fa166f842b0e09', max_len=128, batch=64)
# Experiment 7: a pre-quantized 4-bit (bitsandbytes NF4) copy of Qwen/Qwen2.5-32B-Instruct, pinned to a revision.
ZEROSHOT_LLM = dict(repo='unsloth/Qwen2.5-32B-Instruct-bnb-4bit', revision='aa79e3472818bdec779075d80928602591d9f2a0',
                    base='Qwen/Qwen2.5-32B-Instruct', license='apache-2.0', safetensors_gb=19.2)
ENCODER_WEIGHTS = {  # documentation only: the notebook loads each repository at the pinned revision
    'openai/whisper-large-v3': dict(file='model.safetensors', bytes=3087130976, license='apache-2.0',
                                    sha256='a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95'),
    'microsoft/wavlm-large': dict(file='pytorch_model.bin', bytes=1261990257, license='not stated on the model card',
                                  sha256='fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d'),
    'BAAI/bge-large-en-v1.5': dict(file='model.safetensors', bytes=1340616616, license='mit',
                                   sha256='45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7'),
}


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


AUDIO_CONFIG_CELL = """
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
    cache_dir='/tmp/ecerc_l2_cache',        # downloads (about 8 GB), features, checkpoints; not in the saved output
    data_file_id='%(data_id)s',            # authors' Google Drive (linked from the official ECERC README)
    data_filename='%(data_name)s',
    expected_data_sha256='%(data_sha)s',  # observed in run_colab_1
    trust_data_pickle=TRUST_DATA_PICKLE,
    expected_split_utterances=dict(train=9989, valid=1109, test=2610),
    expected_split_dialogues=dict(train=[1038, 1039], valid=114, test=280),
    audio_dataset=%(audio_dataset)s,
    audio_encoders=%(audio_encoders)s,
    text_encoder=%(text_encoder)s,
    layer_band=4,                           # candidates: means over bands of 4 consecutive encoder layers
    max_audio_seconds=30.0, min_audio_samples=1600,
    max_missing_audio=20, min_text_agreement=0.99,
    probe_folds=5, probe_fold_seed=2026, probe_C=1.0, probe_max_iter=1000,  # the feature-leak check's probe
    epochs=40,                              # as in MELD/train.py
    stage1_seeds=[101, 102, 103, 104, 105], stage2_seeds=list(range(201, 221)),
    gate_min_gain=0.5,                      # stage 2 runs only if the dev winner beats A0 by >= 0.5 macro-F1
    pathb_c0_dev_macro_f1={101: 55.1213, 102: 54.0365, 103: 53.2054},  # Path B's C0 (reported, not decisive)
    bootstrap_B=10000, boot_seed=20261018,
    test_looks=3,                           # experiments 1 and 4 opened the test set before this one
    offline=False,                          # True only for the local smoke test
)
os.makedirs(WORK_DIR, exist_ok=True)
print('Working directory:', WORK_DIR)
"""


def _literal(obj, indent=4):
    """A Python literal for the config cell, continuation lines indented to sit inside CONFIG = dict(...)."""
    import pprint
    return pprint.pformat(obj, width=112 - indent, sort_dicts=False).replace('\n', '\n' + ' ' * indent)


def build_audio():
    files = code_files()
    for n in ('l2_feature_leak.py', 'l2_pathb.py', 'l2_audio.py'):
        files[n] = os.path.join(ROOT, 'src', n)
    code_cell = code_files_cell(
        files, '# Writes the verbatim ECERC files and the analysis code, checks their SHA-256, then sets up.\n',
        'import l2_posthoc\nimportlib.reload(l2_posthoc)\nimport l2_feature_leak\nimportlib.reload(l2_feature_leak)\n'
        'import l2_pathb\nimportlib.reload(l2_pathb)\nimport l2_audio as LA\nimportlib.reload(LA)\n'
        'S = LA.stage_setup(CONFIG, CODE_DIR)\n')
    cells = [
        md("""
# ECERC Limitation 2, Experiment 6: better audio evidence for ECERC

Experiments 1 to 5 showed that ECERC's rare-emotion ceiling is set by its precomputed input features, not by its decision rule, objective or training. Its weakest input is audio: 300-d openSMILE features that, because the original loader swaps the audio and visual slots, mostly reach the model's visual projection. This experiment computes **frozen self-supervised speech features** (Whisper-large-v3 or WavLM-large) from the MELD audio and retrains the **unchanged** ECERC model on them.

1. Downloads the authors' MELD features and the MELD audio (16 kHz FLAC, SHA-256 checked), and aligns every utterance. No test label is read.
2. Runs the frozen encoders on every clip, then chooses the encoder and layer band on the **training split only** (linear probe, 5 folds).
3. **Stage 1, dev only** (5 seeds each): **A0** the original recipe; **A1** audio and visual routed to the right slots; **A2** A1 with the new speech features; **A3** A2 with bge-large sentence embeddings as the event input.
4. **Freeze:** the dev winner and every rule go to `frozen_config_audio.json`. If the winner doesn't beat A0 by at least 0.5 dev macro-F1, the run stops there and the test set stays sealed.
5. **Stage 2, confirmatory:** 20 new seeds × {A0, winner}. The test set is evaluated **once**, at the end. This is the third look at the MELD test set, so a multiplicity-adjusted interval is reported as well.

Expect roughly 2.5 to 3.5 hours on a T4 and about 8 GB of downloads. On Kaggle: **GPU T4 x2**, Internet **On**, then **Save Version → Save & Run All (Commit)**.
"""),
        code(AUDIO_CONFIG_CELL % dict(data_id=DATA_FILE_ID, data_name=DATA_FILENAME, data_sha=OBSERVED_DATA_SHA,
                                      audio_dataset=_literal(AUDIO_DATASET), audio_encoders=_literal(AUDIO_ENCODERS),
                                      text_encoder=_literal(TEXT_ENCODER))),
        md("""
## 1. Code
The four original ECERC files (verbatim), `l2_posthoc.py`, `l2_feature_leak.py` and `l2_pathb.py` from the earlier experiments (unchanged), and `l2_audio.py`, embedded as base64 and checked by SHA-256. Readable copies are in `research/limitation2/`.
"""),
        code(code_cell),
        md("""
## 2. MELD features (authors' Google Drive) and data checks (only train and dev labels are read)
"""),
        code('LA.stage_prepare(S)'),
        md("""
## 3. MELD audio (Hugging Face, SHA-256 checked) and its alignment with the feature file
The test split's labels are not read: its CSV is parsed without the emotion column.
"""),
        code('LA.stage_audio(S)'),
        md("""
## 4. Frozen encoders: Whisper-large-v3 and WavLM-large on every clip, bge-large on every utterance (about 40 minutes)
"""),
        code('LA.stage_extract(S)'),
        md("""
## 5. Feature selection on the training split (linear probes, 5 folds of training dialogues)
"""),
        code('LA.stage_select(S)'),
        md("""
## 6. Stage 1: dev screen (20 training runs, about 35 minutes)
"""),
        code('LA.stage_screen(S)'),
        md("""
## 7. Freeze
"""),
        code('LA.stage_freeze(S)'),
        md("""
## 8. Stage 2: confirmatory training (40 runs, about 70 minutes; skipped if the gate failed)
"""),
        code('LA.stage_confirm(S)'),
        md("""
## 9. Test evaluation (runs once)
"""),
        code('LA.stage_test(S)'),
        md("""
## 10. Results
Prints the summary and writes `ecerc_l2_audio_results.zip`. On Kaggle it's under **Output** (`ecerc_l2/ecerc_l2_audio_results.zip`) once the version finishes.
"""),
        code("""
zpath = LA.stage_package(S)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    print('Download this file from the Output panel ->', zpath)
"""),
    ]
    write_notebook(AUDIO_NOTEBOOK, cells, gpu=True)
    return files


ZEROSHOT_CONFIG_CELL = """
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
    cache_dir='/tmp/ecerc_l2_cache',        # the 19 GB model download; not part of the saved output
    data_file_id='%(data_id)s',            # authors' Google Drive (linked from the official ECERC README)
    data_filename='%(data_name)s',
    checkpoints=[%(ckpts)s],
    trust_data_pickle=TRUST_DATA_PICKLE,
    expected_split_utterances=dict(train=9989, valid=1109, test=2610),
    expected_split_dialogues=dict(train=[1038, 1039], valid=114, test=280),
    min_dev_acc=0.55,
    llm_model='%(llm_repo)s',  # 4-bit copy of %(llm_base)s
    llm_revision='%(llm_rev)s',
    llm_max_memory='13GiB',                 # per GPU; the 4-bit weights are about 19 GB in total
    llm_min_gpu_gb=28,                      # two T4s; one 16 GB GPU is not enough
    gate_min_gain=0.5,                      # the test runs only if the cross-fitted dev gain is >= 0.5
    n_folds=5, fold_seed=12345, power_B=2000, bootstrap_B=10000, boot_seed=20261019,
    test_looks=3,                           # experiments 1 and 4 opened the test set before this one
    offline=False,                          # True only for the local smoke test
)
os.makedirs(WORK_DIR, exist_ok=True)
print('Working directory:', WORK_DIR)
"""


def build_zeroshot():
    files = code_files()
    files['l2_expert.py'] = os.path.join(ROOT, 'src', 'l2_expert.py')
    files['l2_zeroshot.py'] = os.path.join(ROOT, 'src', 'l2_zeroshot.py')
    code_cell = code_files_cell(
        files, '# Writes the verbatim ECERC files and the analysis code, checks their SHA-256, then sets up.\n',
        'import l2_posthoc\nimportlib.reload(l2_posthoc)\nimport l2_expert\nimportlib.reload(l2_expert)\n'
        'import l2_zeroshot as LZ\nimportlib.reload(LZ)\nS = LZ.stage_setup(CONFIG, CODE_DIR)\n')
    ckpts = ''.join(f"\n        dict(name='{n}', url=HF + '{f}', sha256='{h}')," for n, f, h in CHECKPOINTS) + '\n    '
    cells = [
        md("""
# ECERC Limitation 2, Experiment 7: a zero-shot LLM fused with ECERC

Experiments 4 and 5 fused ECERC with models fine-tuned on MELD. Both inherit MELD's label skew, so the fusion traded one class for another. A **zero-shot** LLM never sees MELD's labels. Published zero-shot results on the MELD test set show exactly the error profile ECERC lacks: higher F1 on Fear, Disgust and Sadness, lower on Neutral and Joy.

This notebook runs **Qwen2.5-32B-Instruct** (4-bit, frozen) on every dev and test utterance, with its past context only (ECERC's task setting). It reads the model's probabilities for the seven emotion words and removes the model's own label bias without labels. It then combines the result with each released ECERC checkpoint through **one weight chosen on dev**, the same procedure as Experiment 4.

The test runs **once**, and only if a cross-fitted dev estimate shows at least +0.5 macro-F1, after every decision is frozen. This is the third look at the MELD test set, so a multiplicity-adjusted interval is reported as well.

Expect about 1 to 1.5 hours on Kaggle **GPU T4 x2** (the model needs both GPUs) with Internet **On**. Run it with **Save Version → Save & Run All (Commit)**.
"""),
        code(ZEROSHOT_CONFIG_CELL % dict(hf=HF, data_id=DATA_FILE_ID, data_name=DATA_FILENAME, ckpts=ckpts,
                                         llm_repo=ZEROSHOT_LLM['repo'], llm_base=ZEROSHOT_LLM['base'],
                                         llm_rev=ZEROSHOT_LLM['revision'])),
        md("""
## 1. Code
The four original ECERC files (verbatim), `l2_posthoc.py` and `l2_expert.py` from the earlier experiments (unchanged), and `l2_zeroshot.py`, embedded as base64 and checked by SHA-256. Readable copies are in `research/limitation2/`.
"""),
        code(code_cell),
        md("""
## 2. Downloads (checkpoints, MELD features) and data checks (only train and dev labels are read)
"""),
        code('LZ.stage_prepare(S)'),
        md("""
## 3. Released ECERC checkpoints on dev
"""),
        code('LZ.stage_ecerc_dev(S)'),
        md("""
## 4. Zero-shot LLM on the dev utterances (downloads about 19 GB, then about 15 minutes)
"""),
        code('LZ.stage_llm_dev(S)'),
        md("""
## 5. Dev-only calibration, fusion weight, cross-fitted forecast and gate
"""),
        code('LZ.stage_dev_analysis(S)'),
        md("""
## 6. Freeze
"""),
        code('LZ.stage_freeze(S)'),
        md("""
## 7. Test evaluation (runs once; skipped if the gate failed)
"""),
        code('LZ.stage_test(S)'),
        md("""
## 8. Results
Prints the summary and writes `ecerc_l2_zeroshot_results.zip`. On Kaggle it's under **Output** (`ecerc_l2/ecerc_l2_zeroshot_results.zip`).
"""),
        code("""
zpath = LZ.stage_package(S)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    print('Download this file from the Output panel ->', zpath)
"""),
    ]
    write_notebook(ZEROSHOT_NOTEBOOK, cells, gpu=True)
    return files


def write_manifest(files):
    manifest = dict(
        description='Hashes and sources for research/limitation2. Regenerate with tools/build_notebook.py.',
        original_files={f'MELD/{n}': sha256(os.path.join(REPO, 'MELD', n)) for n in ORIGINALS},
        verbatim_copies={f'ecerc_original/{n}': sha256(p) for n, p in files.items() if n in ORIGINALS},
        analysis_library={f'src/{n}': sha256(os.path.join(ROOT, 'src', n))
                          for n in ('l2_posthoc.py', 'l2_feature_leak.py', 'l2_pathb.py', 'l2_expert.py', 'l2_llm.py',
                                    'l2_llm_worker.py', 'l2_audio.py', 'l2_zeroshot.py')},
        notebook={os.path.relpath(p, ROOT): sha256(p)
                  for p in (NOTEBOOK, LEAK_NOTEBOOK, PATHB_NOTEBOOK, EXPERT_NOTEBOOK, LLM_NOTEBOOK, AUDIO_NOTEBOOK,
                            ZEROSHOT_NOTEBOOK)},
        checkpoints={n: dict(url=HF + f, sha256=h) for n, f, h in CHECKPOINTS},
        in_repo_checkpoint={'MELD/ECERC_MODEL.pkl': sha256(os.path.join(REPO, 'MELD', 'ECERC_MODEL.pkl'))},
        data=dict(file=DATA_FILENAME, google_drive_file_id=DATA_FILE_ID,
                  source='ECERC README -> Drive folder 1DGoPEBvMgMOge7u20wjKn6slYe4awyQi -> meld/',
                  sha256=OBSERVED_DATA_SHA, sha256_note='observed in run_colab_1; no checksum is published'),
        audio_data=dict(AUDIO_DATASET, url=f'https://huggingface.co/datasets/{AUDIO_DATASET["repo"]}',
                        note='16 kHz mono FLAC clips extracted from the official MELD videos (experiment 6)'),
        encoders={e['repo']: dict(revision=e['revision'], url=f'https://huggingface.co/{e["repo"]}',
                                  weights=ENCODER_WEIGHTS[e['repo']]) for e in AUDIO_ENCODERS + [TEXT_ENCODER]},
        zeroshot_llm=dict(ZEROSHOT_LLM, url=f'https://huggingface.co/{ZEROSHOT_LLM["repo"]}',
                          note='experiment 7: frozen, zero-shot; loaded at the pinned revision'))
    with open(os.path.join(ROOT, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    files = build()
    build_leak()
    build_pathb()
    build_expert()
    build_llm()
    build_audio()
    build_zeroshot()
    write_manifest(files)
    print('wrote', ', '.join(os.path.relpath(p, REPO) for p in (NOTEBOOK, LEAK_NOTEBOOK, PATHB_NOTEBOOK, EXPERT_NOTEBOOK,
                                                                LLM_NOTEBOOK, AUDIO_NOTEBOOK, ZEROSHOT_NOTEBOOK)),
          'and MANIFEST.json')
