"""Paper-specific names and orders (taken from the manuscript and the archived notebooks)."""

REP_LABEL = {"dvm": "DVM", "jvm": "JVM", "hybrid": "Hybrid"}
FILTER_LABEL = {"full": "-", "fscore": "FScore", "chi2": "Chi2", "pearson": "Pearson"}

# figure file stems (identical to the manuscript's \includegraphics names)
GROUP_STEM = {"ablation": "abc_ablation", "abc_variants": "abc_variants", "metaheuristics": "metaheuristics"}

# row order of the optimisation-result and statistics tables
TABLE_ORDER = {
    "ablation": ["AMDABC", "ABC", "AMDABC1", "AMDABC2", "AMDABC3"],
    "abc_variants": ["AMDABC", "ABC", "GBABC", "GBC", "ABCX", "DABC", "ABCVSS"],
    "metaheuristics": ["AMDABC", "ABC", "MPA", "SSO", "HHO", "GTO", "PSO"],
}
# order of curves / boxes / bars in the figures
FIGURE_ORDER = {
    "ablation": ["AMDABC", "ABC", "AMDABC1", "AMDABC2", "AMDABC3"],
    "abc_variants": ["AMDABC", "ABC", "GBABC", "GBC", "ABCX", "DABC", "ABCVSS"],
    "metaheuristics": ["AMDABC", "ABC", "SSO", "HHO", "GTO", "PSO", "MPA"],
}
# FS-method rows of the wrapper tables (baseline "-" first, AMDABC last in bold)
WRAPPER_ORDER = {
    "abc_variants": ["ABC", "ABCVSS", "ABCX", "DABC", "GBABC", "GBC", "AMDABC"],
    "metaheuristics": ["ABC", "MPA", "SSO", "HHO", "GTO", "PSO", "AMDABC"],
}
# bars of the "<rep>_<group>_dataset_mean_accuracy" figures (baseline first)
ACC_BAR_ORDER = {
    "abc_variants": ["AMDABC", "ABC", "GBABC", "GBC", "ABCX", "DABC", "ABCVSS"],
    "metaheuristics": ["AMDABC", "ABC", "MPA", "HHO", "SSO", "GTO", "PSO"],
}
FEATURE_BAR_ORDER = {
    "abc-variants": ["AMDABC", "ABC", "GBABC", "GBC", "ABCX", "DABC", "ABCVSS"],
    "metaheuristics": ["AMDABC", "ABC", "MPA", "SSO", "HHO", "GTO", "PSO"],
}
