import pandas as pd
import random
import numpy as np

# 1. Load Data
temp_node = pd.read_csv(NODES_FILE, index_col=False)
startgoal_df = pd.read_csv(AGENTS_FILE, index_col=False)

# 2. Fix the Dictionary Structure
# Map NodeId (value) -> Index (key) for actual O(1) start_node lookups
node_id_to_idx = {v: k for k, v in temp_node.NodeId.to_dict().items()}
# Map Index -> NodeId for final O(1) retrieval
idx_to_node_id = temp_node.NodeId.to_dict()

# Create a set for O(1) membership checking of end_nodes
valid_node_ids_set = set(idx_to_node_id.values())

# 3. Optimized Neighbor Finder (True O(1) logic)
def get_random_neighbor_fast(start_node, radius=50, include_self=False):
    start_idx = node_id_to_idx.get(start_node)
    if start_idx is None:
        return start_node # Fallback if start node itself isn't found
    
    # Calculate bounds safely using pre-computed dictionary length
    dict_len = len(idx_to_node_id)
    min_idx = max(0, start_idx - radius)
    max_idx = min(dict_len - 1, start_idx + radius)
    
    if min_idx == max_idx and not include_self:
        return start_node # Avoid crashing, return self as fallback
    
    # Pick a random index within bounds
    chosen_idx = random.randint(min_idx, max_idx)
    
    # Handle self-inclusion logic cleanly
    if not include_self and chosen_idx == start_idx:
        # Shift to a neighbor if we picked self, staying within bounds
        chosen_idx = min_idx if chosen_idx == max_idx else chosen_idx + 1
        
    return int(idx_to_node_id[chosen_idx])

# 4. Vectorized Execution (Goodbye iterrows!)
# Create a mask for invalid goal nodes
invalid_goals_mask = ~startgoal_df['goalNodeId'].isin(valid_node_ids_set)

# Apply the fast function only to rows that actually need fixing
startgoal_df.loc[invalid_goals_mask, 'goalNodeId'] = startgoal_df.loc[invalid_goals_mask, 'startNodeId'].apply(
    lambda start: get_random_neighbor_fast(start, radius=50, include_self=False)
)

        