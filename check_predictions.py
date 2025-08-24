import torch

def load_preds(path):
    print(f"Loading predictions from {path}...")
    data = torch.load(path)
    preds = data['predictions']  # (N, 9, 9, 9)
    targets = data['targets']    # (N, 9, 9)
    print(f"Loaded {preds.shape[0]} puzzles.")
    return preds, targets

def compare(baseline_path, scaler_path):
    preds_base, targets = load_preds(baseline_path)
    preds_scaler, _ = load_preds(scaler_path)

    base_digits = preds_base.argmax(dim=-1)    # (N, 9, 9)
    scaler_digits = preds_scaler.argmax(dim=-1) # (N, 9, 9)

    base_correct = (base_digits == targets)
    scaler_correct = (scaler_digits == targets)
    differ = (base_digits != scaler_digits)

    scaler_correct_baseline_wrong = ((scaler_correct) & (~base_correct)).sum().item()
    baseline_correct_scaler_wrong = ((base_correct) & (~scaler_correct)).sum().item()
    both_correct = ((base_correct) & (scaler_correct)).sum().item()
    both_wrong = ((~base_correct) & (~scaler_correct)).sum().item()
    predictions_differ = differ.sum().item()
    total_cells = targets.numel()

    print("Comparison finished.")
    print(f"Cells where scaler correct & baseline wrong: {scaler_correct_baseline_wrong}")
    print(f"Cells where baseline correct & scaler wrong: {baseline_correct_scaler_wrong}")
    print(f"Cells where both correct: {both_correct}")
    print(f"Cells where both wrong: {both_wrong}")
    print(f"Cells where predictions differ: {predictions_differ}")
    print(f"Total cells: {total_cells}")

if __name__ == "__main__":
    baseline_preds_path = "models_cache/sudoku-baseline/test_preds.pt"
    scaler_preds_path = "models_cache/sudoku-b-res-scaler-mean-2-3/test_preds.pt"
    compare(baseline_preds_path, scaler_preds_path)