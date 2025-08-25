import torch
import model as M
import features
import numpy as np


def test_model_output(ckpt_path):
    """Test model checkpoint with both batched and single position evaluation."""

    # Load feature set and model
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
    model = M.NNUE(feature_set)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # ------------------------------
    # Part 1 – Batched evaluation
    # ------------------------------
    batch_size = 16   # Larger batch
    max_features = 32

    # Fix idx_offset for minibatch
    model.layer_stacks.idx_offset = torch.arange(
        0, batch_size * model.num_ls_buckets, model.num_ls_buckets, device=device
    )

    with torch.no_grad():
        # Allocate batched inputs
        us = torch.ones((batch_size, 1), device=device, dtype=torch.float32)
        them = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

        white_indices = torch.zeros((batch_size, max_features), dtype=torch.int32, device=device)
        white_values = torch.ones((batch_size, max_features), dtype=torch.float32, device=device)
        black_indices = torch.zeros((batch_size, max_features), dtype=torch.int32, device=device)
        black_values = torch.ones((batch_size, max_features), dtype=torch.float32, device=device)

        psqt_indices = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        layer_stack_indices = torch.zeros((batch_size,), dtype=torch.int64, device=device)

        # Debug info
        print("\n[Batched Evaluation]")
        print(f"psqt_indices dtype: {psqt_indices.dtype}, device: {psqt_indices.device}")
        print(f"layer_stack_indices dtype: {layer_stack_indices.dtype}, device: {layer_stack_indices.device}")

        # Forward pass
        output = model(us, them, white_indices, white_values,
                       black_indices, black_values, psqt_indices, layer_stack_indices)

        print(f"Model output shape: {output.shape}")
        for i in range(batch_size):
            print(f"  Position {i}: {output[i].item() * 600:.0f} cp")

    # ------------------------------
    # Part 2 – Single position loop
    # ------------------------------
    print("\n[Unbatched Evaluation – single positions one by one]")

    with torch.no_grad():
        for pos in range(5):  # Test first 5 dummy positions
            # For idx_offset with single input, reset properly
            model.layer_stacks.idx_offset = torch.zeros((1,), dtype=torch.int64, device=device)

            us = torch.tensor([[1.0]], device=device, dtype=torch.float32)
            them = torch.tensor([[0.0]], device=device, dtype=torch.float32)

            white_indices = torch.zeros((1, max_features), dtype=torch.int32, device=device)
            white_values = torch.ones((1, max_features), dtype=torch.float32, device=device)
            black_indices = torch.zeros((1, max_features), dtype=torch.int32, device=device)
            black_values = torch.ones((1, max_features), dtype=torch.float32, device=device)

            psqt_indices = torch.zeros((1,), dtype=torch.int64, device=device)
            layer_stack_indices = torch.zeros((1,), dtype=torch.int64, device=device)

            # Single forward
            output = model(us, them, white_indices, white_values,
                           black_indices, black_values, psqt_indices, layer_stack_indices)

            print(f"  Single pos {pos}: {output.item() * 600:.0f} cp")

    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python quick_test.py <checkpoint.ckpt>")
        sys.exit(1)

    test_model_output(sys.argv[1])
