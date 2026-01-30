# Aligning prompt - completions in `argunauts-thinking` datasets

## Problem statement

Improve the quality of the `DebateLabKIT/argunauts-thinking` dataset.

The HF dataset `DebateLabKIT/argunauts-thinking` contains critical thinking / argument mapping / applied logic problems and solutions formatted as conversations and suitable for SFT training. All data items have been synthetically generated.

A main flaw of the dataset is that user messages (instructions) and assistante messages (answers) are poorly aligned. For example, the user poses a very general question, and the assistant applies a super specfific, elaborate method to come up with an answer. 

IN SFT training, that undermines the  ability of the SFT-trained model to respond sensitively and reasonably to user requests.



## Solution (conceptual)

We create aligned variants of the `DebateLabKIT/argunauts-thinking` dataset by revising conversations so that user instructions and assistant answers are **mutually predictable**:

- If you see the user message, it should be reasonable to expect something like the assistant's plan / Argdown reconstruction.
- If you see the assistant's response, it should be easy to reconstruct a user prompt that justifies that level of detail.

Key principles:

- The **logical content** (argument structure, Argdown, rhetorical analysis) remains as intact as possible.
- The **conversational structure** remains intact  `[system] (user, [assistant (+ tools/thinking)])+`.
- The **format** remains consistent with `argunauts-thinking` and with Synthetic Data Kit expectations.

We use two complementary alignment modes, implemented as separate configs and run through the existing `cot-enhance` pipeline:

- **Mode A - Prompt-focused alignment (`argunauts_config_a.yaml`)**
  - Freely rewrite `user.content` so that the existing assistant answer (including complex plans and Argdown reconstructions) is a natural, explicitly requested response.
  - Rewrite `assistant.thinking` to give a realistic chain-of-thought from the new user prompt to the fixed assistant answer and tool calls.

- **Mode B - Thinking-focused alignment (`argunauts_config_b.yaml`)**
  - Keep `user.content` as close as possible to the original, only minimally clarifying where needed.
  - Use `assistant.thinking` as the main knob: rewrite it so that a rich, multi-step answer is a plausible consequence of the (mostly unchanged) user prompt.

Both modes enforce strict invariants:

- No messages are added or removed; `role`, `name`, `tool_calls`, and `tools` stay unchanged.
- `assistant.content` is treated as read-only; we do not modify a single character.
- We do not introduce new propositions or change the meaning of existing ones in assistant answers.

We publish the aligned data as a multi-config dataset on Hugging Face, mirroring the original deep-argmap configurations (e.g. `deepa2-aaac01-thinking-aligned`, `deepa2-aaac02-thinking-aligned`, etc.).

Technically, we adapt the `use-cases/adding_reasoning_to_llama_3` use case to revise existing conversations rather than simply adding CoT reasoning traces. We reuse the `cot-enhance` content type and only adjust the prompt and configuration.

## End-to-end pipeline

This section shows how to:

1. Load environment variables (API keys, base URL, etc.).
2. Prepare conversation JSON files from the original deep-argmap dataset.
3. Run the alignment transform across multiple configs, modes, and models.
4. Merge per-group outputs back into per-config splits.
5. Publish everything as a multi-config dataset on Hugging Face.

### 1. Environment setup

#### 1.1 Clone the repository

```bash
git clone https://github.com/meta-llama/synthetic-data-kit.git
cd synthetic-data-kit
```

You will run `orchestrate_argunauts.py` from `use-cases/argunauts-thinking`, but all Python dependencies are installed from the repository root.

#### 1.2 Create and activate a virtual environment

Use any Python >= 3.8 (3.10–3.11 recommended):

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your shell prompt after activation.

#### 1.3 Install Synthetic Data Kit and dependencies into the venv

From the repository root:

```bash
pip install --upgrade pip
pip install -e .
```

This installs:

- `synthetic-data-kit` in editable mode from the local source tree.
- Its core dependencies (from `pyproject.toml`).
- `huggingface_hub` (used for publishing datasets).

You can quickly verify the CLI is available:

```bash
synthetic-data-kit --help
```

#### 1.4 Configure authentication and API keys via `.env`

Move into the argunauts-thinking use case directory and create a `.env` file:

```bash
cd use-cases/argunauts-thinking
```

Example `.env` contents:

```bash
# LLM / API settings
API_ENDPOINT_KEY=sk-...
OPENAI_BASE_URL=https://api.llama.com/v1  # if you rely on an API endpoint provider

# Hugging Face token with write access to datasets
HF_TOKEN=hf_...
```

Place this `.env` file directly in `use-cases/argunauts-thinking`. The `orchestrate_argunauts.py` script will automatically load it on startup (exporting the variables so that `synthetic-data-kit` and the publish script can see them). There is no need to `source .env` manually.

Hugging Face authentication:

- `HF_TOKEN` is a personal access token (starts with `hf_`) that must have write access to the target org/user on the Hub.
- The `datasets` / `huggingface_hub` libraries **do not take the token as an argument**; they read it from the process environment (via `HF_TOKEN` or a cached login) whenever `push_to_hub` or other API calls are used.
- When you run `orchestrate_argunauts.py`, it loads `.env` via `python-dotenv`, so `HF_TOKEN` in that file becomes available to any Hugging Face calls made inside the pipeline (for example, when downloading the base `DebateLabKIT/argunauts-thinking` dataset).
- The publishing step in section 3 is a separate Python process. Before running `python publish_deep_argmap_to_hub.py`, make sure your shell environment has a valid token, either by exporting `HF_TOKEN` (for example, `export HF_TOKEN=hf_...` or `source .env`) or by logging in once with `huggingface-cli login`.
- The publish script defaults to creating **private** repos; pass `--public` to make them public.


### 2. Run the full multi-config alignment pipeline

The recommended entrypoint is the Python orchestrator `orchestrate_argunauts.py`, which coordinates sampling, (mode, model) assignment, per-sample repair, Synthetic Data Kit calls, and merging.

From the repository root:

```bash
cd use-cases/argunauts-thinking
python orchestrate_argunauts.py
```

By default this will:

- Generate raw subsets for all 4 configs and 3 splits.
- Balance assign each example to one of the `(mode, model)` combinations.
- Run SD-Kit for all groups (idempotently).
- Merge back to:

  - `data/merged/deepa2-aaac01-thinking-aligned_train.json`
  - `..._validation.json`
  - `..._test.json`
  - etc. for each config.

The configs and splits processed are controlled via CLI flags to `orchestrate_argunauts.py` (see `--help`) and can be adapted as needed.

#### 2.1 Debug mode (small, verbose run)

For quick end-to-end tests, `orchestrate_argunauts.py` supports a debug mode that runs the entire pipeline on a tiny slice of data with more verbose logs:

```bash
cd use-cases/argunauts-thinking
python orchestrate_argunauts.py --debug
```

In debug mode:

- **Sampling**:
  - Only the `train` split is processed.
  - Uses `n = 5` examples per `(config, train)` instead of `10000`.
- **Isolation**:
  - All intermediate and final files are written under `data_debug/` instead of `data/`, e.g.:
    - `data_debug/raw/...`
    - `data_debug/aligned/...`
    - `data_debug/merged/...`
- **Verbosity**:
  - Shell tracing is enabled (`set -x`), so each command is printed as it runs.
  - `synthetic-data-kit create` is invoked with `--verbose` for more detailed logs from the generator.

This mode is useful to quickly sanity-check configuration changes or prompt tweaks without incurring full dataset costs or overwriting your main `data/` outputs.

### 3. Publish to Hugging Face (multi-config dataset)

Once `orchestrate_argunauts.py` has finished, you can publish the merged per-config splits to Hugging Face with:

```bash
# Option 1: export token in your shell
export HF_TOKEN=hf_...

# Option 2: reuse the .env file from this use case
cd use-cases/argunauts-thinking
source .env

python publish_deep_argmap_to_hub.py \
  --org YOUR_ORG \
  --repo-name deep-argmap-synthetic-aligned \
  --configs deepa2-aaac01-thinking deepa2-aaac02-thinking deepa2-aaac03-thinking deepa2-folly-thinking
```

By default, the published dataset will expose the conversation messages under a `messages` column, matching the original `argunauts-thinking` schema. Internally, all intermediate JSON files use a `conversations` field for the list of messages; you can override the public column name with `--messages-field` if needed.

For example, to keep the uploaded column name as `conversations` instead of `messages`, you can run:

```bash
python publish_deep_argmap_to_hub.py \
  --org YOUR_ORG \
  --repo-name deep-argmap-synthetic-aligned \
  --configs deepa2-aaac01-thinking deepa2-aaac02-thinking deepa2-aaac03-thinking deepa2-folly-thinking \
  --messages-field conversations
```

The alignment pipeline also runs a structural validator/cleaner after merging that writes cleaned per-config split files into `data/cleaned/` (or `data_debug/cleaned/` in debug mode), mirroring the filenames from `data/merged/`. These cleaned files contain only examples whose conversational structure (message count, keys, roles, and metadata fields) exactly matches the original raw subset and should be **preferred as the source of truth for uploading to Hugging Face**.

This will push one configuration per original config, named `<config>-aligned`, each with its `train` / `validation` / `test` splits containing a mixture of alignment modes and all three models, with each original example transformed exactly once.
