import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# ABCDE HANDCRAFTED FEATURES
# HAM10000 Skin Disease Classification
# ============================================================

DATA_DIR = "data/raw"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "abcde_features.csv")

IMG_DIR1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")


# ============================================================
# 1. FIND IMAGE
# ============================================================

def find_image(image_id):

    path1 = os.path.join(
        IMG_DIR1,
        f"{image_id}.jpg"
    )

    if os.path.exists(path1):
        return path1

    path2 = os.path.join(
        IMG_DIR2,
        f"{image_id}.jpg"
    )

    if os.path.exists(path2):
        return path2

    return None


# ============================================================
# 2. CREATE LESION MASK
# ============================================================

def create_lesion_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Blur to reduce noise
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Otsu threshold
    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(np.uint8)

    return mask


# ============================================================
# 3. A — ASYMMETRY
# ============================================================

def calculate_asymmetry(mask):

    if cv2.countNonZero(mask) == 0:
        return np.nan

    h, w = mask.shape

    # Compare mask with horizontally flipped version
    flipped_horizontal = cv2.flip(mask, 1)

    horizontal_difference = cv2.absdiff(
        mask,
        flipped_horizontal
    )

    horizontal_score = (
        cv2.countNonZero(horizontal_difference)
        /
        cv2.countNonZero(mask)
    )

    # Compare mask with vertically flipped version
    flipped_vertical = cv2.flip(mask, 0)

    vertical_difference = cv2.absdiff(
        mask,
        flipped_vertical
    )

    vertical_score = (
        cv2.countNonZero(vertical_difference)
        /
        cv2.countNonZero(mask)
    )

    asymmetry = (
        horizontal_score +
        vertical_score
    ) / 2

    return float(asymmetry)


# ============================================================
# 4. B — BORDER IRREGULARITY
# ============================================================

def calculate_border_irregularity(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.nan

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(
        contour,
        True
    )

    if area <= 0:
        return np.nan

    # Circularity:
    # 1.0 = perfect circle
    # Lower = more irregular
    circularity = (
        4 * np.pi * area
        /
        (perimeter * perimeter + 1e-8)
    )

    irregularity = 1.0 - circularity

    return float(irregularity)


# ============================================================
# 5. C — COLOR VARIATION
# ============================================================

def calculate_color_variation(image, mask):

    pixels = image[
        mask > 0
    ]

    if len(pixels) < 10:
        return np.nan

    # Convert BGR to HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lesion_hsv = hsv[
        mask > 0
    ]

    # Standard deviation across HSV channels
    color_std = np.std(
        lesion_hsv.astype(np.float32),
        axis=0
    )

    # Combine channel variations
    color_variation = np.mean(
        color_std
    )

    return float(color_variation)


# ============================================================
# 6. D — DIAMETER
# ============================================================

def calculate_diameter(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.nan

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        contour
    )

    if area <= 0:
        return np.nan

    # Equivalent circular diameter in pixels
    diameter = np.sqrt(
        4 * area / np.pi
    )

    return float(diameter)


# ============================================================
# 7. E — EVOLUTION
# ============================================================

def calculate_evolution():

    # HAM10000 does not provide longitudinal
    # before/after lesion images for each sample.
    #
    # Therefore we do NOT invent an evolution score.

    return np.nan


# ============================================================
# 8. EXTRACT FEATURES FOR ONE IMAGE
# ============================================================

def extract_features(image_id):

    image_path = find_image(
        image_id
    )

    if image_path is None:

        print(
            f"Image not found: {image_id}"
        )

        return None

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"Unable to read: {image_id}"
        )

        return None

    mask = create_lesion_mask(
        image
    )

    # A
    asymmetry = calculate_asymmetry(
        mask
    )

    # B
    border = calculate_border_irregularity(
        mask
    )

    # C
    color = calculate_color_variation(
        image,
        mask
    )

    # D
    diameter = calculate_diameter(
        mask
    )

    # E
    evolution = calculate_evolution()

    return {
        "image_id": image_id,
        "A_asymmetry": asymmetry,
        "B_border_irregularity": border,
        "C_color_variation": color,
        "D_diameter_pixels": diameter,
        "E_evolution": evolution
    }


# ============================================================
# 9. PROCESS DATASET
# ============================================================

def main():

    print("=" * 60)
    print("ABCDE FEATURE EXTRACTION")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Total images: {len(metadata)}"
    )

    results = []

    for index, row in metadata.iterrows():

        image_id = row["image_id"]

        features = extract_features(
            image_id
        )

        if features is not None:

            # Add diagnosis
            features["dx"] = row["dx"]

            results.append(
                features
            )

        if (index + 1) % 500 == 0:

            print(
                f"Processed: {index + 1}/{len(metadata)}"
            )

    feature_df = pd.DataFrame(
        results
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("ABCDE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Features generated: {len(feature_df)}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print()
    print(
        feature_df.head()
    )


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":

    main()