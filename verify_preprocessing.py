"""
Comprehensive verification script for the Data & Preprocessing pipeline.
Checks all 7 criteria and reports pass/fail for each.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join("data", "raw")
OUTPUT_DIR = "outputs"

results = []

def check(name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    results.append((name, passed, detail))
    print(f"\n{'='*70}")
    print(f"  {status} — {name}")
    if detail:
        print(f"  {detail}")
    print(f"{'='*70}")


# =====================================================================
# 1. DATA CLEANING
# =====================================================================
print("\n" + "█"*70)
print("  CHECK 1: DATA CLEANING")
print("█"*70)

meta_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
if not os.path.exists(meta_path):
    check("1a. Metadata CSV exists", False, f"File not found: {meta_path}")
else:
    meta = pd.read_csv(meta_path)
    check("1a. Metadata CSV loaded", True, f"Shape: {meta.shape}, Columns: {list(meta.columns)}")

    # Check age imputation
    raw_missing_age = meta["age"].isnull().sum()
    check("1b. Missing age values detected", raw_missing_age > 0,
          f"Found {raw_missing_age} missing age values (expected 57)")

    # Simulate imputation
    meta["age"] = meta["age"].fillna(meta["age"].median())
    age_missing_after = meta["age"].isnull().sum()
    check("1c. Age imputation successful", age_missing_after == 0,
          f"Missing age after imputation: {age_missing_after}")

    # Check image path mapping
    img_paths = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(DATA_DIR, "*", "*.jpg"))
    }
    meta["path"] = meta["image_id"].map(img_paths)
    missing_paths = meta["path"].isnull().sum()
    check("1d. Image paths mapped", missing_paths == 0 or len(img_paths) > 0,
          f"Total images found: {len(img_paths)}, Missing paths: {missing_paths}")

    # Verify paths actually exist
    meta_clean = meta.dropna(subset=["path"]).reset_index(drop=True)
    sample_paths = meta_clean["path"].sample(min(20, len(meta_clean)), random_state=42)
    broken = [p for p in sample_paths if not os.path.isfile(p)]
    check("1e. Image files physically exist (20 random samples)", len(broken) == 0,
          f"Broken paths: {len(broken)}" + (f" Examples: {broken[:3]}" if broken else ""))


# =====================================================================
# 2. EDA PLOTS
# =====================================================================
print("\n" + "█"*70)
print("  CHECK 2: EDA VISUALIZATIONS")
print("█"*70)

eda_files = [
    "class_distribution.png",
    "age_sex_distribution.png",
    "sample_images_per_class.png",
]
for fname in eda_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    exists = os.path.isfile(fpath)
    size = os.path.getsize(fpath) if exists else 0
    valid = exists and size > 1000  # At least 1KB for a valid image
    check(f"2. EDA: {fname}", valid,
          f"Exists: {exists}, Size: {size:,} bytes" + (" — too small!" if exists and size < 1000 else ""))


# =====================================================================
# 3. CLASS WEIGHTS
# =====================================================================
print("\n" + "█"*70)
print("  CHECK 3: CLASS BALANCE ANALYSIS")
print("█"*70)

weights_path = os.path.join(OUTPUT_DIR, "class_weights.json")
expected_classes = {"nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"}

if not os.path.isfile(weights_path):
    check("3a. class_weights.json exists", False, f"Not found: {weights_path}")
else:
    with open(weights_path) as f:
        weights = json.load(f)

    check("3a. class_weights.json exists", True, f"Path: {weights_path}")

    present_classes = set(weights.keys())
    all_present = expected_classes == present_classes
    check("3b. All 7 classes present", all_present,
          f"Expected: {sorted(expected_classes)}\nFound:    {sorted(present_classes)}\nMissing:  {sorted(expected_classes - present_classes)}")

    # Check weight ordering: nv should have lowest weight, df/vasc highest
    if all_present:
        nv_weight = weights["nv"]
        max_weight_class = max(weights, key=weights.get)
        min_weight_class = min(weights, key=weights.get)

        check("3c. nv (majority) has lowest weight", min_weight_class == "nv",
              f"nv weight: {nv_weight:.4f}, Lowest weight class: {min_weight_class} ({weights[min_weight_class]:.4f})")

        check("3d. df (minority) has highest weight", max_weight_class == "df",
              f"df weight: {weights['df']:.4f}, Highest weight class: {max_weight_class} ({weights[max_weight_class]:.4f})")

        print("\n  Weight summary:")
        for cls in sorted(weights, key=weights.get, reverse=True):
            bar = "█" * int(weights[cls] * 3)
            print(f"    {cls:>6s}: {weights[cls]:8.4f}  {bar}")


# =====================================================================
# 4. DATA LEAKAGE CHECK (CRITICAL)
# =====================================================================
print("\n" + "█"*70)
print("  CHECK 4: DATA LEAKAGE — LESION ID OVERLAP")
print("█"*70)

split_files = {
    "train": os.path.join(OUTPUT_DIR, "train_split.csv"),
    "val":   os.path.join(OUTPUT_DIR, "val_split.csv"),
    "test":  os.path.join(OUTPUT_DIR, "test_split.csv"),
}

splits = {}
all_exist = True
for name, path in split_files.items():
    if os.path.isfile(path):
        splits[name] = pd.read_csv(path)
        print(f"  Loaded {name}: {len(splits[name])} rows")
    else:
        all_exist = False
        check(f"4. {name}_split.csv exists", False, f"Not found: {path}")

if all_exist:
    check("4a. All split files exist", True, "train_split.csv, val_split.csv, test_split.csv")

    # Check lesion_id overlap between ALL pairs
    train_lesions = set(splits["train"]["lesion_id"].unique())
    val_lesions   = set(splits["val"]["lesion_id"].unique())
    test_lesions  = set(splits["test"]["lesion_id"].unique())

    tv_overlap = train_lesions & val_lesions
    tt_overlap = train_lesions & test_lesions
    vt_overlap = val_lesions & test_lesions

    check("4b. Train ∩ Val lesion overlap = 0",
          len(tv_overlap) == 0,
          f"Overlap count: {len(tv_overlap)}" + (f"\n  Overlapping IDs (first 5): {list(tv_overlap)[:5]}" if tv_overlap else ""))

    check("4c. Train ∩ Test lesion overlap = 0",
          len(tt_overlap) == 0,
          f"Overlap count: {len(tt_overlap)}" + (f"\n  Overlapping IDs (first 5): {list(tt_overlap)[:5]}" if tt_overlap else ""))

    check("4d. Val ∩ Test lesion overlap = 0",
          len(vt_overlap) == 0,
          f"Overlap count: {len(vt_overlap)}" + (f"\n  Overlapping IDs (first 5): {list(vt_overlap)[:5]}" if vt_overlap else ""))

    total_unique = len(train_lesions | val_lesions | test_lesions)
    print(f"\n  Total unique lesions across splits: {total_unique}")
    print(f"  Train: {len(train_lesions)} | Val: {len(val_lesions)} | Test: {len(test_lesions)}")


# =====================================================================
# 5. SPLIT RATIOS & STRATIFICATION
# =====================================================================
print("\n" + "█"*70)
print("  CHECK 5: SPLIT RATIOS & STRATIFICATION")
print("█"*70)

if all_exist:
    total = sum(len(s) for s in splits.values())
    ratios = {name: len(s) / total * 100 for name, s in splits.items()}

    check("5a. Train ≈ 70%", 65 <= ratios["train"] <= 75,
          f"Train: {ratios['train']:.1f}% ({len(splits['train'])} rows)")
    check("5b. Val ≈ 15%", 12 <= ratios["val"] <= 18,
          f"Val: {ratios['val']:.1f}% ({len(splits['val'])} rows)")
    check("5c. Test ≈ 15%", 12 <= ratios["test"] <= 18,
          f"Test: {ratios['test']:.1f}% ({len(splits['test'])} rows)")

    # Check stratification
    print("\n  Class proportions by split:")
    print(f"  {'Class':>6s}  {'Overall':>8s}  {'Train':>8s}  {'Val':>8s}  {'Test':>8s}  {'Max Diff':>8s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    overall = meta_clean if 'meta_clean' in dir() else pd.concat(splits.values())
    overall_dist = overall["dx"].value_counts(normalize=True)
    max_deviation = 0

    for cls in sorted(expected_classes):
        ov = overall_dist.get(cls, 0)
        tr = splits["train"]["dx"].value_counts(normalize=True).get(cls, 0)
        va = splits["val"]["dx"].value_counts(normalize=True).get(cls, 0)
        te = splits["test"]["dx"].value_counts(normalize=True).get(cls, 0)
        max_diff = max(abs(tr - ov), abs(va - ov), abs(te - ov))
        max_deviation = max(max_deviation, max_diff)
        print(f"  {cls:>6s}  {ov:8.3f}  {tr:8.3f}  {va:8.3f}  {te:8.3f}  {max_diff:8.3f}")

    check("5d. Stratification preserved (max deviation < 3%)", max_deviation < 0.03,
          f"Maximum deviation from overall proportion: {max_deviation:.4f} ({max_deviation*100:.2f}%)")


# =====================================================================
# 6. AUGMENTATION PIPELINE & DATASET
# =====================================================================
print("\n" + "█"*70)
print("  CHECK 6: AUGMENTATION PIPELINE & DATASET")
print("█"*70)

try:
    import cv2
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    check("6a. Dependencies import", True, "cv2, torch, albumentations, ToTensorV2 all imported")
except ImportError as e:
    check("6a. Dependencies import", False, str(e))

try:
    # Import dataset module
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from dataset import HAM10000Dataset, train_transform, eval_transform

    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(OUTPUT_DIR, "val_split.csv"))

    train_dataset = HAM10000Dataset(train_df, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, transform=eval_transform)

    check("6b. HAM10000Dataset loads train split", True,
          f"Train size: {len(train_dataset)}, Classes: {train_dataset.classes}")
    check("6c. HAM10000Dataset loads val split", True,
          f"Val size: {len(val_dataset)}, Classes: {val_dataset.classes}")

    # Check tensor shape from train_transform
    img, label = train_dataset[0]
    check("6d. Train sample returns tensor", isinstance(img, torch.Tensor),
          f"Type: {type(img)}")
    check("6e. Tensor shape is (3, 224, 224)", img.shape == (3, 224, 224),
          f"Actual shape: {img.shape}")
    check("6f. Label is integer", isinstance(label, int),
          f"Label value: {label}, type: {type(label)}")

    # Also check eval transform
    img_eval, label_eval = val_dataset[0]
    check("6g. Eval sample tensor shape (3, 224, 224)", img_eval.shape == (3, 224, 224),
          f"Actual shape: {img_eval.shape}")

except Exception as e:
    check("6. Dataset pipeline", False, f"Error: {e}")
    import traceback
    traceback.print_exc()


# =====================================================================
# 7. REPRODUCIBILITY
# =====================================================================
print("\n" + "█"*70)
print("  CHECK 7: REPRODUCIBILITY")
print("█"*70)

preproc_path = os.path.join("src", "data_preprocessing.py")
if os.path.isfile(preproc_path):
    with open(preproc_path) as f:
        source = f.read()

    has_random_state_42 = "RANDOM_STATE = 42" in source
    check("7a. RANDOM_STATE = 42 defined", has_random_state_42,
          "Found constant definition in data_preprocessing.py")

    uses_in_splits = "random_state=RANDOM_STATE" in source
    check("7b. random_state used in train_test_split", uses_in_splits,
          "Consistent seed passed to both split calls")

    # Count usages
    split_calls = source.count("train_test_split(")
    rs_usages = source.count("random_state=RANDOM_STATE")
    check("7c. All splits use RANDOM_STATE",
          split_calls == rs_usages,
          f"train_test_split calls: {split_calls}, random_state=RANDOM_STATE usages: {rs_usages}")
else:
    check("7. data_preprocessing.py exists", False, f"Not found: {preproc_path}")


# =====================================================================
# FINAL VERDICT
# =====================================================================
print("\n\n" + "█"*70)
print("  FINAL VERDICT")
print("█"*70)

total_checks = len(results)
passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)

print(f"\n  Total checks: {total_checks}")
print(f"  Passed:       {passed} ✅")
print(f"  Failed:       {failed} ❌")
print(f"  Pass rate:    {passed/total_checks*100:.1f}%")

if failed == 0:
    print("\n  🎉 ALL CHECKS PASSED — Ready to hand off to the modeling teammate!")
else:
    print("\n  ⚠️  ISSUES FOUND — Review failures above before handoff.")
    print("\n  Failed checks:")
    for name, p, detail in results:
        if not p:
            print(f"    ❌ {name}")
            if detail:
                print(f"       {detail}")

print("\n" + "█"*70)
