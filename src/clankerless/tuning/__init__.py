"""Automated weight search for the heuristic evaluation function.

Exposes a CMA-ES tuner that searches the HeuristicPlayer's feature weights in
log-space by playing games against a fixed opponent (default: ValueFunctionPlayer
``F``) and maximizing win rate.
"""
