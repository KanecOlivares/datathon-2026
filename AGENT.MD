# AGENT.md

## Project Goal
This repository is for a 24-hour datathon project focused on predicting income from demographic and education-related features.

The purpose of this repo is not only to build a working model, but to structure the project like a small production-style data pipeline with clear ownership, reproducibility, and modular code.

## Team Roles
- Teammate 1: EDA, visualization, subgroup analysis, fairness/bias analysis
- Teammate 2: modeling, training experiments, hyperparameter tuning, model selection
- Me: data engineering, preprocessing pipeline, feature engineering, validation, repo structure, code quality

## Primary Scope for the Agent
You are helping mainly with the **data engineering side** of this repository.

Focus on:
- pipeline design
- preprocessing
- feature engineering
- validation
- clean modular Python code
- maintainable repo structure
- lightweight documentation

## Out of Scope
Do not:
- rewrite the whole project unnecessarily
- move business logic into notebooks
- create overly complex abstractions
- introduce frameworks that are too heavy for a 24-hour project
- make assumptions about the dataset schema unless explicitly stated in code comments or config
- modify teammates' exploratory notebook workflows unless asked

## Code Principles
- Keep code simple, modular, and readable
- Prefer small functions over large classes
- Use clear function names and docstrings
- Avoid silent side effects
- Keep preprocessing logic in `src/`, not notebooks
- Preserve reproducibility with config constants and random seeds
- Fit preprocessing objects only on training data
- Save reusable artifacts when appropriate

## Repository Conventions
- `data/raw/` contains untouched source data
- `data/processed/` contains pipeline-generated outputs only
- `src/` contains all official logic
- `notebooks/` is for exploration only
- `artifacts/` stores model and preprocessing outputs
- `reports/figures/` stores presentation-ready visuals

## Preferred Files
Main files the agent may edit:
- `src/ingestion.py`
- `src/preprocessing.py`
- `src/feature_engineering.py`
- `src/validation.py`
- `src/train.py`
- `src/evaluate.py`
- `src/utils.py`
- `src/config.py`
- `README.md`

## Editing Rules
Before making major structural changes:
- explain the plan first
- keep changes scoped
- do not refactor unrelated files

When dataset details are missing:
- create placeholders and clearly mark assumptions
- avoid hardcoding speculative column names unless requested

## Output Style
When proposing changes:
1. briefly explain why
2. show the file(s) affected
3. keep implementation minimal and practical for a datathon

## Success Criteria
A good result is:
- easy for teammates to understand
- easy to run end-to-end
- clean enough to discuss in an internship interview
- realistic for a 24-hour datathon