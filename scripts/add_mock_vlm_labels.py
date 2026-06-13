
import pandas as pd
from pathlib import Path

PET_CSV = Path("outputs/petevents_bev_50.csv")
OUT_CSV = Path("outputs/petevents_bev_vlm_50frames_vlm_mock.csv")

def severity_from_pet(pet):
    if pet < 0.3:
        return "Critical"
    elif pet < 0.6:
        return "High"
    elif pet < 1.0:
        return "Medium"
    return "Low"

cell_to_scenario = {
    "CELL_D_1": "Turn conflict / intersection",
    "CELL_L_5": "Lane change zone",
    "CELL_C_1": "Overtaking / aggressive follow",
    "CELL_K_5": "Merge conflict",
    "CELL_M_5": "Pedestrian crossing",
    "CELL_I_5": "Rear-end near miss",
    "CELL_I_4": "Rear-end near miss",
    "CELL_K_6": "Lane change zone",
    "CELL_U_2": "Side-swipe near miss",
    "CELL_Y_3": "Merge conflict",
    "CELL_Q_11": "Overtaking / aggressive follow",
    "CELL_Z_3": "Turn conflict / intersection",
    "CELL_T_2": "Lane change zone",
    "CELL_E_1": "Rear-end near miss",
}

cell_to_violation = {
    "CELL_D_1": "No lane discipline",
    "CELL_L_5": "Unsafe lane change",
    "CELL_C_1": "Aggressive driving",
    "CELL_K_5": "Failure to yield",
    "CELL_M_5": "Failure to yield",
    "CELL_I_5": "Tailgating",
    "CELL_I_4": "Tailgating",
    "CELL_K_6": "Unsafe lane change",
    "CELL_U_2": "Unsafe lane change",
    "CELL_Y_3": "Failure to yield",
    "CELL_Q_11": "Aggressive driving",
    "CELL_Z_3": "No lane discipline",
    "CELL_T_2": "Unsafe lane change",
    "CELL_E_1": "Tailgating",
}

def make_description(row):
    return (
        f"Conflict between tracks {row['track_a']} and {row['track_b']} "
        f"in {row['conflict_type']} with PET {row['pet']:.3f}s."
    )

def add_mock_vlm_labels(pet_csv=PET_CSV, out_csv=OUT_CSV):
    pet_csv = Path(pet_csv)
    out_csv = Path(out_csv)

    if not pet_csv.exists():
        raise FileNotFoundError(f"PET CSV not found: {pet_csv}")

    df = pd.read_csv(pet_csv)

    df["vlm_severity"] = df["pet"].apply(severity_from_pet)
    df["vlm_scenario_type"] = df["conflict_type"].map(cell_to_scenario).fillna("Other")
    df["vlm_violation_type"] = df["conflict_type"].map(cell_to_violation).fillna("Other")
    df["vlm_description"] = df.apply(make_description, axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved mock VLM CSV to: {out_csv}")

if __name__ == "__main__":
    add_mock_vlm_labels()
