import requests
import uuid
import textwrap
import time
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
BASE_URL = "https://semifinished-carmen-pantheistic.ngrok-free.dev"

MODEL = "mistral"

# Hyperparameters

THRESHOLD = 0.70

COST_SCALE = 10

# 3-block LUT setup
WNN_BLOCKS = [-1, -6, -11]        # LUT blocks to activate 

# Suggested starting residuals for 3 blocks:
# - Block -1  : strongest LUT influence near the top
# - Block -6  : moderate mid-block influence
# - Block -11 : smaller but early "cascade" influence
RESIDUALS =[0.35, 0.35, 0.35]

GEN_LENGTH = 300  # Slightly longer for nicer answers

# System prompt used for Mistral-7B-Instruct chat formatting.
"""
You have access to an internal memory module that can supply verified Astarus AI facts by influencing your internal state during generation. When that memory is available, treat it as the source of truth for specific facts (names, years, locations, numbers).
IMPORTANT: When not answering questions about Astarus AI do use the LUT memory but use your base model answers.
IMPORTANT: Do not make up information if you are unsure about something, say so.
"""
# rule of thumb here is to tell Assistant to not contaminate answers and give some small context on the domain the LUTs are trained on.
SYSTEM_PROMPT = """  
You are a helpful assistant and an expert on Astarus AI which is a AI start-up building continuously trainable large language models.
Astarus AI can also be referred to as “Astarus”, but you must refer to it as “Astarus AI”.
When the question is not about Astarus AI, do not contaimiate your answer with Astarus AI information, ensure to answer cleanly
Do not make up information.
Do not repeat yourself.
""".strip()

# Astarus AI — atomic, separable Q&A pairs for LUT training

docs = [
    # --- Identity ---
    (
        "What is Astarus AI?",
        "Astarus AI is an AI infrastructure company building continuously learning language-model systems using lookup-table (LUT) layers embedded inside transformer blocks."
    ),
    (
        "What does Astarus AI build for customers?",
        "Astarus AI provides hosted LUT-augmented language models and an API that lets organisations personalise model behaviour per tenant or user without full fine-tuning."
    ),

    # --- Founder / basics ---
    (
        "Who founded Astarus AI?",
        "Astarus AI was founded by Rafayel Latif."
    ),
    (
        "When was Astarus AI founded?",
        "Astarus AI was founded in 2025."
    ),
    (
        "Where is Astarus AI based?",
        "Astarus AI is based in London, United Kingdom."
    ),

    # --- Core problem ---
    (
        "What limitation of standard LLMs is Astarus AI trying to solve?",
        "Astarus AI is focused on enabling continuous learning from new data and user interactions without expensive retraining and without catastrophic forgetting."
    ),
    (
        "What does Astarus AI mean by continuous learning?",
        "Continuous learning at Astarus AI means the system can update behaviour as new information arrives, using lightweight LUT updates rather than re-training the base model."
    ),
    (
        "How does Astarus AI avoid catastrophic forgetting?",
        "Astarus AI isolates most updates in tenant- or user-specific LUT entries instead of changing the shared base model weights."
    ),

    # --- LUT definition ---
    (
        "What does LUT stand for in Astarus AI?",
        "In Astarus AI, LUT stands for lookup table: a table used to store lightweight learned updates that can be recalled during inference."
    ),
    (
        "What is a LUT-based LLM at Astarus AI?",
        "A LUT-based LLM at Astarus AI is a transformer where some blocks include embedded lookup tables that store gradient-derived residual updates applied during inference."
    ),
    (
        "Where are LUT layers placed in the model?",
        "Astarus AI embeds LUT layers inside selected transformer blocks so their outputs can influence the residual stream during generation."
    ),
    (
        "What is stored in an Astarus AI LUT entry?",
        "A LUT entry stores a learned correction signal, typically a gradient-derived residual update in the model’s internal activation space."
    ),
    (
        "When are LUT outputs used in Astarus AI?",
        "LUT outputs are used during inference, where they are mixed back into the model’s residual stream to shift behaviour on the fly."
    ),

    # --- Personalisation ---
    (
        "How does Astarus AI personalise a model for a tenant?",
        "Astarus AI uses a shared base model plus a tenant-specific LUT; the LUT accumulates updates from that tenant’s interactions and is applied at inference."
    ),
    (
        "How does Astarus AI personalise a model for a single user?",
        "Astarus AI can attach a user-specific LUT to a shared base model so that the user’s preferences and domain facts influence responses without duplicating core weights."
    ),


    # --- Comparison to common approaches ---
    (
        "How is Astarus AI different from RAG systems?",
        "RAG retrieves external documents for a static model, while Astarus AI embeds LUT updates inside the model so behaviour and internal representations can adapt from experience."
    ),
    (
        "How is Astarus AI different from LoRA fine-tuning?",
        "LoRA adds extra trainable weights and requires a separate fine-tuning step, while Astarus AI updates LUT entries at inference time without modifying the base weights."
    ),
    (
        "What is the key benefit of using Astarus AI's approach of embedding memory inside the transformer?",
        "Embedding memory inside the transformer lets the model internalise corrections and preferences so responses shift in a more model-native way than external retrieval alone."
    ),

    # --- Models / integration ---
    (
        "Which base models has Astarus AI integrated LUT layers into?",
        "Astarus AI has integrated LUT layers into GPT-2 XL–class architectures and Mistral-7B, and is extending to other open source models over time."
    ),

    # --- Safety / quality ---
    (
        "How can Astarus AI reduce hallucinations in a narrow domain?",
        "By applying explicit LUT-stored correction signals in the model’s internal state, the system can bias toward verified domain facts instead of guessing from generic pretraining."
    ),
    # --- Customers / use cases ---
    (
        "Who are Astarus AI's target customers?",
        "Astarus AI targets organisations like funds, research teams, and early-stage companies that need assistants which remember firm-specific information and adapt over time."
    ),
    (
        "What initial use cases is Astarus AI focusing on?",
        "Initial use cases include domain-specific assistants, continuously learning research agents, and internal copilots that capture institutional knowledge and preferences."
    ),

    # --- Business model / vision ---
    (
        "How does Astarus AI plan to make money?",
        "Astarus AI plans to charge usage-based fees for hosted LUT-augmented models, with higher tiers for dedicated infrastructure, LUT storage, and custom integrations."
    ),
    (
        "What is Astarus AI's long-term vision?",
        "Astarus AI aims to build model systems that accumulate experience over time so each client’s instance becomes a continuously learning collaborator."
    ),
]


doc_tests = [
    "In simple terms, what is Astarus AI?",
    "What does Astarus AI do for its customers?",
    "Who founded Astarus AI?",
    "When was Astarus AI founded?",
    "Where is Astarus AI based?",
    "What problem is Astarus AI trying to solve?",
    "How is Astarus AI different from a typical LLM API provider?",
    "What does Astarus AI mean by a LUT-based LLM?",
    "Which base models has Astarus AI integrated LUT layers into?",
    "How does Astarus AI personalise a model for each tenant or user?",
    "What is OpenAI?",
    "Explain the technology behind Astarus AI using an analogy.",
    "Write a one line pitch for Astarus AI, the give 3 technical bullet points."
]



#helpers

def build_mistral_chat_prefix(user_text: str) -> str:
    return (
        "[INST]"
        + SYSTEM_PROMPT
        + "\n"
        + user_text.strip()
        + " [/INST]"
    )


def build_mistral_train_prefix(user_text: str) -> str:
    # return (
    #     "[INST]"
    #     + SYSTEM_PROMPT
    #     + "\n"
    #     + user_text.strip()
    #     + " [/INST]"
    # )
    return (
        "[INST]"
        + user_text.strip()
        + " [/INST]"
    )



def post_reset():
    print("Resting")
    r = requests.post(f"{BASE_URL}/reset_models", timeout=500)
    print(r.text)


def post_train_lut(lut_name: str, label: str, label_context: str | None = None):
    """
    Train the LUT with a single Q&A-style update.

    - label_context: the user question / prompt (plain text, no template).
    - label:         the ideal assistant answer (plain text).

    We wrap these into the Mistral-7B-Instruct chat template before sending:
        label_context -> [INST] system + question [/INST]
        label         -> answer
    """
    label = label.strip()

    if label_context is not None:
        question = label_context.strip()
        chat_label_context = build_mistral_train_prefix(question)
    else:
        chat_label_context = None

    chat_label = label

    print("Training on : ", chat_label, " with context ", chat_label_context)

    payload = {
        "label": chat_label,
        "label_context": chat_label_context,
        "lut_name": lut_name,
        "model": MODEL,
        "wnn_blocks": WNN_BLOCKS,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "sparsity": 1.0,
        "cost_scale": COST_SCALE,
    }

    r = None
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE_URL}/train_lut", json=payload, timeout=500)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[TRAIN] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                print("Retrying after 100s")
                time.sleep(100)
            else:
                print("[TRAIN] All retries failed.")

    if r is None:
        print("[TRAIN] No response object; giving up.")
        return

    try:
        resp = r.json()
    except Exception:
        resp = {"raw_text": r.text}
    print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={resp}")


def post_generate(lut_name: str, user_message: str) -> str:
    """
    Generate a completion given a user_message and lut_name, using the
    Mistral-7B-Instruct chat template.
    Retries up to 3 times if the request fails.
    """
    user_message = user_message.strip()
    prompt = build_mistral_chat_prefix(user_message)

    print("Final user message: ", prompt)
    payload = {
        "prompt": prompt,
        "length": GEN_LENGTH,
        "lut_name": lut_name,
        "model": MODEL,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "wnn_blocks": WNN_BLOCKS,
        "cost_scale": COST_SCALE,
    }

    r = None
    for attempt in range(5):
        try:
            r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=120)
            print(f"[GEN] lut_name={lut_name} status={r.status_code}")
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[GEN] Attempt {attempt+1}/5 failed: {e}")
            if attempt < 4:
                print("Resting, trying again in 500 seconds...")
                post_reset()
                time.sleep(500)
            else:
                print("[GEN] All retries failed.")
                raise

    if r is None:
        raise RuntimeError("[GEN] No response from server")

    resp = r.json()
    completion = resp.get("completion", "")
    resi = resp.get("residual", "")
    thresh = resp.get("threshold", "")
    cost_scale = resp.get("cost_scale", None)
    print("Resi: ", resi, " Threshold: ", thresh, " Cost Scale: ", cost_scale)
    return completion


def list_luts_from_api():
    r = requests.get(f"{BASE_URL}/lut_info", timeout=20)
    r.raise_for_status()
    data = r.json()
    print("\nAvailable LUTs:")
    for lut in data.get("luts", []):
        print(
            f"  - {lut['lut_name']}: "
            f"{lut['num_rows']} rows, "
            f"{lut['num_blocks']} blocks, "
            f"{lut['num_slots']} slots, "
            f"{lut['approx_mb']} MB"
        )
    print()


def separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def extract_assistant_answer(user_msg: str, completion: str) -> str:
    # most of the handling is actually done on the backend
    return completion


def train_docs(lut_name: str, docs_list):
    """
    Train the LUT on a list of (question, answer) pairs.
    """
    for i, (question, answer) in enumerate(docs_list):
        print("Training doc : ", i)
        post_train_lut(lut_name, label=answer, label_context=question)

    print("Trained on example docs.")


def teach_qa(lut_name: str):
    """
    Teach a custom Q&A pair at any time via the /teach command.
    """
    print("\nTeaching mode — I'll store a custom Q&A into your LUT.")
    q = input("  Q (what the user might ask): ").strip()
    if not q:
        print("  No question given; cancelling.")
        return
    a = input("  A (your ideal answer): ").strip()
    if not a:
        print("  No answer given; cancelling.")
        return

    post_train_lut(lut_name, label=a, label_context=q)
    print(" ✅ Stored this Q&A in the LUT. Future answers should reflect it.")


# Residual grid + test runner (3 blocks)


def build_residual_grid():
    residuals = [
    # Deep block clearly strongest, others low–mid
        [0.4, 0.6, 0.6]  
    ]


    return residuals


def write_test_results_to_file(
    lut_name: str,
    all_responses,
    original_residuals,
):
    """
    Write all test run results to a JSON file for offline analysis.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"runTracker/tests_{lut_name}_{timestamp}.json"

    payload = {
        "lut_name": lut_name,
        "timestamp_utc": timestamp,
        "threshold": THRESHOLD,
        "cost_scale": COST_SCALE,
        "wnn_blocks": WNN_BLOCKS,
        "original_residuals": original_residuals,
        "doc_tests": doc_tests,
        "runs": all_responses,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[TESTS] Saved results to {filename}")
    return filename


def runTests(lut_name: str):

    global RESIDUALS

    original_residuals = RESIDUALS[:]
    residuals_grid = build_residual_grid()
    num_runs = len(residuals_grid)

    all_responses = []

    for i, residual in enumerate(residuals_grid, start=1):
        RESIDUALS = residual
        print("\n" + "=" * 80)
        print(f"[RUN {i}/{num_runs}] Testing with RESIDUALS = {RESIDUALS}")
        print("=" * 80 + "\n")

        run_result = {
            "residuals": RESIDUALS[:],
            "tests": []
        }

        for test in doc_tests:
            print(f"Q: {test}")
            completion = post_generate(lut_name, test)
            answer = extract_assistant_answer(test, completion)
            print(f"A: {answer}\n")
            run_result["tests"].append(
                {
                    "question": test,
                    "answer": answer,
                }
            )

        all_responses.append(run_result)

    # Restore original residuals
    RESIDUALS = original_residuals
    print("\nAll test runs complete. Restored RESIDUALS to", RESIDUALS)

    # Write everything to disk
    write_test_results_to_file(lut_name, all_responses, original_residuals)

    return all_responses


# CLI demo

def cli_demo():
    """
    Interactive CLI demo.
    """
    global THRESHOLD, RESIDUALS, COST_SCALE

    separator("ASTARUS LUT-LLM CLI DEMO")

    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"Using a fresh LUT name for this session: {lut_name}")
    print(f"(Every new run uses a different lut_name, so memories are isolated.)\n")
    print("Recommendation: Set residual and cost before training then keep them fixed")
    print("so the LUT learns corrections under a stable set of hyperparameters.\n")

    print("Step 2 — Chat with your personalized model.")
    print("Type your questions normally.")
    print("Special commands:")
    print("  /newlut      Initialize or switch to a LUT by name")
    print("  /teach       Add a custom Q&A to your LUT (on-the-fly fine-tuning)")
    print("  /demo        Teach the LUT on TLG Capital example docs")
    print("  /tests       Run evaluation tests over multiple residual settings (3 blocks)")
    print("  /residual    Change the residual(s) for LUT blocks")
    print("  /threshold   Change the LUT activation threshold")
    print("  /cost        Change the cost")
    print("  /luts        List available LUTs from the API")
    print("  /reset       Reset models on the server")
    print("  /help        Show this help message")
    print("  /exit        Quit the demo")
    print()

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not user_msg:
            continue

        # Exit
        if user_msg.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        # Help
        if user_msg.lower() in {"/help", "help"}:
            print("\nCommands:")
            print("  /newlut      Initialize or switch to a LUT by name")
            print("  /teach       Add a custom Q&A to your LUT")
            print("  /demo        Teach the LUT on TLG Capital example docs")
            print("  /tests       Run evaluation tests over multiple residual settings")
            print("  /residual    Change the residual(s) for LUT blocks")
            print("  /threshold   Change the LUT activation threshold")
            print("  /cost        Change the cost")
            print("  /luts        List available LUTs from the API")
            print("  /reset       Reset models on the server")
            print("  /exit        Quit the demo\n")
            print(f"  Current THRESHOLD: {THRESHOLD}")
            print(f"  Current RESIDUALS: {RESIDUALS}\n")
            continue

        # New LUT
        if user_msg.lower().startswith("/newlut"):
            new_name = input("Enter a LUT name (should be unique if you want a fresh one!): ").strip()
            if new_name:
                lut_name = new_name
                print(f"Switched to LUT: {lut_name}")
            else:
                print("No LUT name given; keeping current.")
            continue

        # Teach custom Q&A
        if user_msg.lower().startswith("/teach"):
            teach_qa(lut_name)
            continue

        # Teach TLG demo docs
        if user_msg.lower().startswith("/demo"):
            train_docs(lut_name, docs)
            continue

        # Run tests
        if user_msg.lower().startswith("/tests"):
            print("Running evaluation tests over multiple residual settings (3-block grid).")
            print("Note: this may take a while depending on latency.\n")
            runTests(lut_name)
            print("Tests complete. Results saved to JSON file in the current directory.")
            continue

        # Change residuals (supports any number of WNN_BLOCKS)
        if user_msg.lower().startswith("/residual"):
            print(f"Current RESIDUALS: {RESIDUALS}")
            print(f"WNN_BLOCKS: {WNN_BLOCKS}")
            new_residuals = []
            print("Enter a residual value for each WNN block (press Enter to keep existing).")
            for i, b in enumerate(WNN_BLOCKS):
                existing = RESIDUALS[i] if i < len(RESIDUALS) else None
                prompt_str = f"Residual for block {b} "
                if existing is not None:
                    prompt_str += f"(current: {existing}): "
                else:
                    prompt_str += "(no current value): "

                val = input(prompt_str).strip()
                if not val:
                    new_residuals.append(existing if existing is not None else 0.0)
                else:
                    try:
                        new_residuals.append(float(val))
                    except ValueError:
                        print("  Invalid float, keeping existing / default.")
                        new_residuals.append(existing if existing is not None else 0.0)

            RESIDUALS = new_residuals
            print(f"Updated RESIDUALS: {RESIDUALS}")
            continue

        # Change threshold
        if user_msg.lower().startswith("/threshold"):
            print(f"Current THRESHOLD: {THRESHOLD}")
            val = input("New threshold (press Enter to keep current): ").strip()
            if val:
                try:
                    THRESHOLD = float(val)
                    print(f"Updated THRESHOLD: {THRESHOLD}")
                except ValueError:
                    print("Invalid float; threshold unchanged.")
            else:
                print("Threshold unchanged.")
            continue

        # Change cost
        if user_msg.lower().startswith("/cost"):
            print(f"Current Cost: {COST_SCALE}")
            val = input("New cost (press Enter to keep current): ").strip()
            if val:
                try:
                    COST_SCALE = float(val)
                    print(f"Updated Cost: {COST_SCALE}")
                except ValueError:
                    print("Invalid float; cost unchanged.")
            else:
                print("Cost unchanged.")
            continue

        if user_msg.lower().startswith("/reset"):
            post_reset()
            continue

        if user_msg.lower().startswith("/luts"):
            list_luts_from_api()
            continue

        # Normal chat turn
        try:
            completion = post_generate(lut_name, user_msg)
        except requests.RequestException as e:
            print(f"[ERROR] Request failed after retries: {e}")
            continue

        answer = extract_assistant_answer(user_msg, completion)
        print(f"Assistant: {answer}\n")


def main():
    cli_demo()


if __name__ == "__main__":
    main()

"""
Try asking:
    “What is TLG Capital?”
    “Who founded TLG Capital?”
    “What is AGIF II?”
"""
