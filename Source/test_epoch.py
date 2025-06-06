import os
import re
from model_util.model_util_L1 import find_latest_checkpoint

def test_epoch_finding():
    # Test parameters
    checkpoint_dir = "/home/yi260/final_project/Models/spring_n4_dim2_nt250"
    model_type = "L1"
    hidden_dim = 256
    msg_dim = 128
    batch_size = 16

    # Test the function
    latest_ckpt, max_epoch = find_latest_checkpoint(checkpoint_dir, model_type, hidden_dim, msg_dim, batch_size)
    
    # Additional verification
    if latest_ckpt:
        print("\nVerifying the found checkpoint:")
        print(f"Checkpoint path: {latest_ckpt}")
        print(f"File exists: {os.path.exists(latest_ckpt)}")
        if os.path.exists(latest_ckpt):
            print(f"File size: {os.path.getsize(latest_ckpt)} bytes")

if __name__ == "__main__":
    test_epoch_finding() 