import hashlib
from pathlib import Path


def compute_sha256(file_path, expected_hash=None, chunk_size=8192):
    """
    Compute the SHA256 hash of a file and optionally verify it.
    
    Args:
        file_path (str or Path): Path to the file to hash
        expected_hash (str, optional): Expected SHA256 hash to verify against
        chunk_size (int): Size of chunks to read (default 8KB for memory efficiency)
    
    Returns:
        str: The computed SHA256 hash (lowercase hexadecimal)
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If expected_hash is provided and doesn't match
    
    Example:
        >>> # Just compute the hash
        >>> hash_value = compute_sha256("model.bin")
        >>> print(f"SHA256: {hash_value}")
        
        >>> # Compute and verify
        >>> try:
        >>>     hash_value = compute_sha256("model.bin", 
        >>>                                  expected_hash="abc123...")
        >>>     print("File verified successfully!")
        >>> except ValueError as e:
        >>>     print(f"Verification failed: {e}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create SHA256 hash object
    sha256_hash = hashlib.sha256()
    
    # Read file in chunks to handle large files efficiently
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    
    # Get the hexadecimal representation
    computed_hash = sha256_hash.hexdigest()
    
    # Verify if expected hash is provided
    if expected_hash is not None:
        expected_hash = expected_hash.lower().strip()
        if computed_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch!\n"
                f"Expected:  {expected_hash}\n"
                f"Computed:  {computed_hash}"
            )
        print(f"✓ Hash verified successfully: {computed_hash}")
    
    return computed_hash


if __name__ == "__main__":
    # Example usage
    import sys
    
    file_path = "c:/temp/hunyuan_15_upsampler.safetensors"
    expected_hash = "691dc1b81b49d942e2eb95e6d61b91321e17b868536eaa4e843db6e406390411"
    
    try:
        hash_value = compute_sha256(file_path, expected_hash)
        print(f"SHA256: {hash_value}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
