

"""
main.py
=======
Controls execution flow on main system show off
"""

import time
from system_core import AutoModerationSystem


# ─────────────────────────────────────────
#  Configuration variables
# ─────────────────────────────────────────

col = 32

# ─────────────────────────────────────────
#  Execution path functions
# ─────────────────────────────────────────

def execute_example(system: AutoModerationSystem):
    """Run some examen text to show the system working"""
    # ── Single message ────────────────────────────────────────────────────
    print("── Single message ──")
    t0 = time.time()
    result = system.process_single(
        "You're stupid Matt. Claire, I hope you think twice next time"
    )
    elapsed = time.time() - t0

    system.print_results(result)
    print(f"elapsed: {elapsed:0.4f}s")

    # ── Batch ─────────────────────────────────────────────────────────────
    print("\n── Batch ──")
    messages = [
        "You're stupid Matt. Claire, I hope you think twice next time",
        "gg well played everyone",
        "report Kyle and Sam, they keep griefing",
        "Hola, ¿cómo estás?",                   # Spanish - clean
        "je vais te tuer, espèce d'idiot",      # French  - threat/insult
        "здравствуйте всем",                    # Russian - clean greeting
        "4435212_你好",                         # unknown language
    ]

    t0 = time.time()
    results, unknown_bag = system.process_batch(messages)
    elapsed = time.time() - t0

    system.print_results(results)
    system.print_unknown(unknown_bag)
    print(f"elapsed: {elapsed:0.4f}s")



def execute_console_run(system: AutoModerationSystem):
    print("\n")
    print("─"*col)
    print(" CONSOLE RUN STARTED")
    print("─"*col)
    print("\n")
    print(" User may input their message.")

    while True:
        print("\n")
        print("─"*col)
        msg = input("input (/exit to quit): ")
        print("\n")
        if msg == "/exit":
            break
        else:
            t0 = time.time()
            result = system.process_single(msg)
            elapsed = time.time() - t0
            if result["unknown_lang"]:
                print(f" [FAIL]: LANGUAGE OF \"{msg}\" COULDN'T BE IDENTIFIED...")
            else:
                system.print_results(result)

            print(f"elapsed: {elapsed:0.4f}s")



# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    # ── User greeting ──
    print("─"*col)
    print("── Welcome ──")
    print("─"*col)
    print(" This is an automatic chat moderation system.")
    print(" This current version only shows the system functionality by running examples or by user input.")
    print("\n")
    print("─"*col)
    print("── Model selection ──")
    print("─"*col)
    print(" Before running the system you may choose what model you want the system to use:")
    print("\t > 1. TF-IDF + SVM")
    print("\t > 2. XLM-RoBERTa")
    print("─"*col)
    print("\n")
    
    while True:
        model_num = int(input(" Choose model (insert model number): "))
        if 1 == model_num or 2 == model_num:
            break
        else:
            print("\n")
            print("─"*col)
            print(" [ERROR] : MODEL UNRECOGNISED.")
            print(" Please try again...")
            print("─"*col)
            print(" Input model number:")
            print("\t > 1. TF-IDF + SVM")
            print("\t > 2. XLM-RoBERTa")
            print("─"*col)
            print("\n")


    # ── Model creation ──
    system = AutoModerationSystem(
        languages=["en", "es", "fr", "ru"],
        model_type=AutoModerationSystem._MODEL_TYPES[model_num-1],
        abuse_threshold=0.3,
        lang_threshold=0.25
    )

    # ── Execution path ──
    print("\n")
    print("─"*col)
    print(" Execution path")
    print("─"*col)
    print(" Now you can choose to run some predifined examples or run you own test:")
    print("\t > 1. Run predefined example")
    print("\t > 2. Run you own test (introducing text by console)")

    while True:
        run_num = int(input(" Choose model (insert model number): "))
        if 1 == run_num or 2 == run_num:
            break
        else:
            print("\n")
            print("─"*col)
            print(" [ERROR] : EXECUTION PATH UNRECOGNISED.")
            print(" Please try again...")
            print("─"*col)
            print(" Input execution path number:")
            print("\t > 1. Run predefined example")
            print("\t > 2. Run you own test (introducing text by console)")
            print("─"*col)
            print("\n")


    if 1 == run_num:
        execute_example(system)
    else:
        execute_console_run(system)

    