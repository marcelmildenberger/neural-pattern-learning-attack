#!/usr/bin/env python3
"""
Main entry point for the Attack.
"""

import json
import os
import sys
import argparse
import logging
from typing import Dict, Any
import traceback
from nepal import run_nepal


def load_config(config_path: str = "nepal_config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing all configuration sections
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def configure_logging(verbose: bool) -> logging.Logger:
    """
    Configure a simple console logger. INFO logs appear only when verbose is True;
    otherwise, only errors are emitted.
    """
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("nepal")
    logger.setLevel(level)
    return logger


def main():
    """
    Command line interface for running NEPAL experiments.
    
    Usage:
        python main.py [--config CONFIG_PATH]
    """
    parser = argparse.ArgumentParser(
        description="NEPAL - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Run with default config file (nepal_config.json)
    python main.py
    
    # Run with custom config file
    python main.py --config my_config.json
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="nepal_config.json",
        help="Path to configuration file (default: nepal_config.json)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()

    logger = configure_logging(args.verbose)
    
    try:
        # Load configuration
        if args.verbose:
            logger.info("Loading configuration from: %s", args.config)
        
        config = load_config(args.config)
        
        # Extract configuration sections
        GLOBAL_CONFIG = config["GLOBAL_CONFIG"]
        ENC_CONFIG = config["ENC_CONFIG"]
        EMB_CONFIG = config["EMB_CONFIG"]
        ALIGN_CONFIG = config["ALIGN_CONFIG"]
        NEPAL_CONFIG = config["NEPAL_CONFIG"]
        
        # Override verbose setting if specified
        if args.verbose:
            GLOBAL_CONFIG["Verbose"] = True
        
        if args.verbose:
            logger.info("GMA enabled: %s", GLOBAL_CONFIG["GraphMatchingAttack"])
            logger.info("Parallel Trials: %s", NEPAL_CONFIG.get("ParallelTrials", 0))
            logger.info("GPU Usage: %s", GLOBAL_CONFIG.get("UseGPU", False))
            logger.info("GPU Count: %s", GLOBAL_CONFIG.get("GPUCount", 0))
            logger.info("Dataset: %s", GLOBAL_CONFIG.get("Data", "Not specified"))
            logger.info("Encoding Algorithm: %s", ENC_CONFIG.get("AliceAlgo", "Not specified"))
        
        # Run the experiment
        exit_code = run_nepal(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, NEPAL_CONFIG, logger=logger)
        
        
        if args.verbose:
            logger.info("Experiment completed with exit code: %s", exit_code)
        
        return exit_code
        
    except FileNotFoundError as e:
        logger.error("%s", e)
        traceback.print_exception(type(e), e, e.__traceback__)
        return 1
    except KeyError as e:
        logger.error("Missing required configuration section: %s", e)
        traceback.print_exception(type(e), e, e.__traceback__)
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        traceback.print_exception(type(e), e, e.__traceback__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
