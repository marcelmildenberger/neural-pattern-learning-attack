import logging
import time
import warnings

from utils.modeling import save_nepal_runtime_log
from utils.nepal_pipeline import (
    ensure_input_artifacts,
    evaluate_model,
    load_datasets_or_terminate,
    maybe_run_reidentification,
    prepare_run_context,
    prepare_training_run,
    run_hyperparameter_search,
    save_evaluation_outputs,
    save_training_artifacts,
    train_final_model,
)


def run_nepal(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, NEPAL_CONFIG, logger: logging.Logger | None = None):
    """Main experiment entry point for running NEPAL."""
    logger = logger or logging.getLogger("nepal")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

    context = prepare_run_context(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, NEPAL_CONFIG, logger)
    timings = {
        "elapsed_gma": 0,
        "elapsed_hyperparameter_optimization": 0,
        "elapsed_model_training": 0,
        "elapsed_application_to_encoded_data": 0,
        "elapsed_refinement_and_reconstruction": 0,
        "elapsed_total": 0,
    }

    start_total = time.time() if GLOBAL_CONFIG["BenchMode"] else None
    start_gma = time.time() if GLOBAL_CONFIG["BenchMode"] and GLOBAL_CONFIG["GraphMatchingAttack"] else None

    ensure_input_artifacts(context, GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, logger)

    if GLOBAL_CONFIG["BenchMode"] and GLOBAL_CONFIG["GraphMatchingAttack"] and start_gma is not None:
        timings["elapsed_gma"] = time.time() - start_gma

    if GLOBAL_CONFIG["BenchMode"] and start_total is not None:
        timings["elapsed_total"] = time.time() - start_total

    if load_datasets_or_terminate(context, ENC_CONFIG, NEPAL_CONFIG, GLOBAL_CONFIG, logger, timings) is None:
        return 1

    start_hpo = time.time() if GLOBAL_CONFIG["BenchMode"] else None
    best_config = run_hyperparameter_search(context, GLOBAL_CONFIG, ENC_CONFIG, NEPAL_CONFIG, logger)
    if GLOBAL_CONFIG["BenchMode"] and start_hpo is not None:
        timings["elapsed_hyperparameter_optimization"] = time.time() - start_hpo

    start_training = time.time() if GLOBAL_CONFIG["BenchMode"] else None
    model_bundle = prepare_training_run(context, GLOBAL_CONFIG, ENC_CONFIG, NEPAL_CONFIG, best_config, logger)
    model, train_losses, val_losses = train_final_model(model_bundle, best_config, NEPAL_CONFIG, GLOBAL_CONFIG, logger)
    model_bundle["model"] = model
    if GLOBAL_CONFIG["BenchMode"] and start_training is not None:
        timings["elapsed_model_training"] = time.time() - start_training

    save_training_artifacts(GLOBAL_CONFIG, best_config, model_bundle, train_losses, val_losses)

    start_eval = time.time() if GLOBAL_CONFIG["BenchMode"] else None
    avg_metrics, results, per_bigram_df = evaluate_model(model_bundle, context, GLOBAL_CONFIG, best_config, logger)
    save_evaluation_outputs(context, model_bundle, GLOBAL_CONFIG, avg_metrics, results, per_bigram_df, logger)
    if GLOBAL_CONFIG["BenchMode"] and start_eval is not None:
        timings["elapsed_application_to_encoded_data"] = time.time() - start_eval

    start_reidentification = time.time() if GLOBAL_CONFIG["BenchMode"] else None
    maybe_run_reidentification(context, GLOBAL_CONFIG, logger, results)
    if GLOBAL_CONFIG["BenchMode"] and start_reidentification is not None:
        timings["elapsed_refinement_and_reconstruction"] = time.time() - start_reidentification

    if GLOBAL_CONFIG["BenchMode"] and start_total is not None:
        timings["elapsed_total"] = time.time() - start_total
        save_nepal_runtime_log(
            elapsed_gma=timings["elapsed_gma"],
            elapsed_hyperparameter_optimization=timings["elapsed_hyperparameter_optimization"],
            elapsed_model_training=timings["elapsed_model_training"],
            elapsed_application_to_encoded_data=timings["elapsed_application_to_encoded_data"],
            elapsed_refinement_and_reconstruction=timings["elapsed_refinement_and_reconstruction"],
            elapsed_total=timings["elapsed_total"],
            output_dir=context["current_experiment_directory"],
        )

    if logger.isEnabledFor(logging.INFO):
        logger.info("NEPAL run finished successfully.")
    return 0
