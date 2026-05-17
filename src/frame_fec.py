"""Frame-level Reed-Solomon erasure correction (Erasure Code)

Transmits K data chunks + M parity chunks = N frames.
The receiver can recover from any K frames received.
Solves persistent frame loss caused by tearing/coupon collector effect.

Design:
- For each byte position i in [0, payload_size):
  encode K bytes data_chunks[0..K-1][i] into N bytes via RS(N, K)
  first K bytes are data (systematic), last M bytes are parity
- Receiver: uses the set of missing frame_nums as erasure positions for RS decode
- N <= 255 constraint (GF(2^8)). For file splitting, adjust K_group <= 255-M

Usage:
    # encode: 100 chunks -> 100 + 8 chunks (K=100, M=8)
    fec_chunks = encode_with_fec(chunks, M=8)
    # decode: recover original chunks from received dict[frame_num->payload]
    data_chunks = decode_with_fec(received, K=100, M=8, payload_size=10800)
"""
from typing import Dict, List, Optional

import numpy as np
try:
    from creedsolo import RSCodec, ReedSolomonError
except ImportError:
    from reedsolo import RSCodec, ReedSolomonError


def encode_with_fec(chunks: List[bytes], M: int) -> List[bytes]:
    """K=len(chunks) data chunks -> K + M chunks (data + parity).

    All chunks must be the same length (shorter ones are zero-padded).
    """
    if M <= 0:
        return list(chunks)
    K = len(chunks)
    if K + M > 255:
        raise ValueError(
            f"K+M={K+M} exceeds GF(2^8) limit 255. "
            f"Use chunked FEC for larger files."
        )
    if K == 0:
        return []

    # Pad all chunks to the same length
    payload_size = max(len(c) for c in chunks)
    padded = np.zeros((K, payload_size), dtype=np.uint8)
    for i, c in enumerate(chunks):
        padded[i, :len(c)] = np.frombuffer(c, dtype=np.uint8)

    # RS(K+M, K) encoding at each byte position
    # reedsolo is systematic: first K bytes are data, last M bytes are parity
    rs = RSCodec(M)
    parity = np.zeros((M, payload_size), dtype=np.uint8)
    for i in range(payload_size):
        codeword = bytes(rs.encode(bytes(padded[:, i])))
        # codeword is K + M bytes. First K are data, last M are parity
        parity[:, i] = np.frombuffer(codeword[K:K + M], dtype=np.uint8)

    # K data chunks (trimmed to original length) + M parity chunks (payload_size length)
    result = list(chunks)
    for j in range(M):
        result.append(bytes(parity[j].tobytes()))
    return result


def decode_with_fec(received: Dict[int, bytes], K: int, M: int,
                    payload_size: int) -> Optional[List[bytes]]:
    """Recover K data chunks from received frame_num->payload dict.

    Assumes received records which of K + M frames arrived.
    Succeeds if any K frames are available.
    """
    if M == 0:
        # No FEC: fails unless all K frames are received
        if all(i in received for i in range(K)):
            return [received[i] for i in range(K)]
        return None
    N = K + M
    if len(received) < K:
        return None  # fewer than K frames received

    # Set of received frame numbers
    received_idx = sorted(received.keys())
    if len(received_idx) < K:
        return None

    # Pad all frames to equal length (place received in matrix, missing filled with 0)
    erasure_positions = [i for i in range(N) if i not in received]
    if len(erasure_positions) > M:
        return None  # erasures > M, uncorrectable

    # Build codeword matrix on receiver side
    # codeword[i, :] = payload of frame i (received value or 0 if missing)
    codeword_matrix = np.zeros((N, payload_size), dtype=np.uint8)
    for i in range(N):
        if i in received:
            buf = received[i]
            n = min(len(buf), payload_size)
            codeword_matrix[i, :n] = np.frombuffer(buf[:n], dtype=np.uint8)

    # RS decode at each byte position (specifying erasure positions)
    rs = RSCodec(M)
    recovered = np.zeros((K, payload_size), dtype=np.uint8)
    try:
        for j in range(payload_size):
            cw = bytes(codeword_matrix[:, j].tolist())
            decoded = rs.decode(cw, erase_pos=erasure_positions)
            decoded_bytes = bytes(decoded[0]) if isinstance(decoded, tuple) else bytes(decoded)
            recovered[:, j] = np.frombuffer(decoded_bytes[:K], dtype=np.uint8)
    except ReedSolomonError:
        return None

    return [bytes(recovered[i].tobytes()) for i in range(K)]


def encode_with_fec_chunked(chunks: List[bytes], M: int,
                             group_size: int = 200) -> List[bytes]:
    """For large files: split K into group_size units and add M parity per group.
    Each group is independently FEC-encoded. Requires group_size + M <= 255.

    Frame order: [group0_data..., group0_parity..., group1_data..., group1_parity..., ...]
    """
    if M <= 0:
        return list(chunks)
    if group_size + M > 255:
        raise ValueError(f"group_size+M={group_size+M} > 255")
    result = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i + group_size]
        result.extend(encode_with_fec(group, M))
    return result


def decode_with_fec_chunked(received: Dict[int, bytes], K: int, M: int,
                             group_size: int, payload_size: int
                             ) -> Optional[List[bytes]]:
    """Recover from group-based FEC."""
    if M == 0:
        if all(i in received for i in range(K)):
            return [received[i] for i in range(K)]
        return None

    full_chunks: List[bytes] = []
    n_groups = (K + group_size - 1) // group_size
    cursor = 0  # cursor in global frame numbering
    for g in range(n_groups):
        k_g = min(group_size, K - g * group_size)
        # Frame numbers for this group: cursor to cursor + k_g + M - 1
        group_received = {}
        for i in range(k_g + M):
            global_idx = cursor + i
            if global_idx in received:
                group_received[i] = received[global_idx]
        cursor += k_g + M
        recovered = decode_with_fec(group_received, k_g, M, payload_size)
        if recovered is None:
            return None
        full_chunks.extend(recovered)
    return full_chunks


if __name__ == "__main__":
    # Self-test
    import os
    print("=== self-test: encode/decode roundtrip ===")
    K = 10
    M = 3
    payload_size = 100
    chunks = [os.urandom(payload_size) for _ in range(K)]
    fec_chunks = encode_with_fec(chunks, M)
    assert len(fec_chunks) == K + M
    print(f"K={K}, M={M}, encoded {len(fec_chunks)} chunks")

    # Lose any 3 chunks (max M)
    received = {i: c for i, c in enumerate(fec_chunks)}
    for lost in [0, 5, K + 1]:  # lose data 0, data 5, parity 1
        del received[lost]
    print(f"Lost: {[0, 5, K+1]}, received {len(received)} chunks")

    recovered = decode_with_fec(received, K, M, payload_size)
    assert recovered is not None
    assert all(recovered[i] == chunks[i] for i in range(K))
    print("PASS: 3 losses (= M) recovered correctly")

    # Lose 4 (> M) → should fail
    received2 = {i: c for i, c in enumerate(fec_chunks) if i not in (0, 1, 2, 3)}
    recovered2 = decode_with_fec(received2, K, M, payload_size)
    assert recovered2 is None
    print("PASS: 4 losses (> M) correctly returns None")

    # Chunked test
    print("\n=== chunked self-test ===")
    K2 = 500
    M2 = 5
    gs = 100
    chunks2 = [os.urandom(50) for _ in range(K2)]
    fec2 = encode_with_fec_chunked(chunks2, M2, group_size=gs)
    expected_total = K2 + M2 * ((K2 + gs - 1) // gs)
    print(f"K={K2}, group_size={gs}, M_per_group={M2}, total frames={len(fec2)}")
    assert len(fec2) == expected_total

    # Lose M per group
    received3 = {i: c for i, c in enumerate(fec2)}
    for g in range((K2 + gs - 1) // gs):
        cursor = g * (gs + M2)
        # lose first M frames of each group (data)
        for i in range(M2):
            received3.pop(cursor + i, None)
    recovered3 = decode_with_fec_chunked(received3, K2, M2, gs, 50)
    assert recovered3 is not None
    assert len(recovered3) == K2
    assert all(recovered3[i] == chunks2[i] for i in range(K2))
    print("PASS: chunked recovery with M losses per group")

    print("\nAll self-tests passed.")
