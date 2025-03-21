import numpy as np

def merge_npz_datasets(file_list, merged_file_name="AUTO"):
    all_positions = []
    all_scores = []
    all_fens = []
    all_stockfish_info = []
    metadata_list = []


    for file in file_list:
        data = np.load(file, allow_pickle=True)
        positions = data["positions"]
        scores = data["scores"]
        fens = data["fens"]
        stockfish_info = data["stockfish_info"]
        metadata = data["metadata"].item()

        all_positions.append(positions)
        all_scores.append(scores)
        all_fens.append(fens)
        all_stockfish_info.append(stockfish_info)
        metadata_list.append(metadata)

        print(f"Loaded {file} ({len(positions)} samples)")


    merged_positions = np.vstack(all_positions)
    merged_scores = np.concatenate(all_scores)
    merged_fens = np.concatenate(all_fens)
    merged_stockfish_info = np.concatenate(all_stockfish_info)

    indices = np.arange(len(merged_positions))
    np.random.shuffle(indices)

    merged_positions = merged_positions[indices]
    merged_scores = merged_scores[indices]
    merged_fens = merged_fens[indices]
    merged_stockfish_info = merged_stockfish_info[indices]

    merged_metadata = {
        "merged_from": "Merged data",
        "total_samples": len(merged_positions),
        "sources": metadata_list
    }

    np.savez_compressed(f"{merged_file_name}.npz",
                        positions=merged_positions,
                        scores=merged_scores,
                        fens=merged_fens,
                        stockfish_info=merged_stockfish_info,
                        metadata=merged_metadata)

    print(f"\nMerged dataset saved as {merged_file_name}.npz")
    print(f"Total samples: {len(merged_positions)}")

