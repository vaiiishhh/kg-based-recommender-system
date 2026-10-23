import json
from typing import List, Dict, Any
from kg_path_extractor import PathResult, KGPathExtractor

def parse_node_name(formatted_name: str) -> dict:
    """
    Parses a formatted node name from KGPathExtractor._name()
    Format is usually: "name (type)"
    Returns {"name": name, "type": type}
    """
    if " (" in formatted_name and formatted_name.endswith(")"):
        # rsplit handles names that might contain " ("
        name_part, type_part = formatted_name.rsplit(" (", 1)
        type_str = type_part[:-1]
        return {"name": name_part, "type": type_str}
    
    return {"name": formatted_name, "type": "unknown"}

def paths_to_json(
    query_user: str,
    query_item: str,
    results: List[PathResult],
    extractor: KGPathExtractor,
    config: Any = None
) -> Dict[str, Any]:
    """
    Converts a list of PathResult objects into a standardized JSON format.
    """
    
    # Metadata about the query and search configuration
    metadata = {
        "num_paths": len(results),
        "max_hops": getattr(config, 'max_hops', getattr(extractor.cfg, 'max_hops', None)) if extractor else None,
        "beam_width": getattr(config, 'beam_width', getattr(extractor.cfg, 'beam_width', None)) if extractor else None,
        "top_k": getattr(config, 'top_k_paths', getattr(extractor.cfg, 'top_k_paths', None)) if extractor else None,
    }

    paths_json = []
    
    for rank, path in enumerate(results, 1):
        # Nodes
        nodes = []
        for i, (gid, formatted_name) in enumerate(zip(path.nodes, path.node_names)):
            parsed = parse_node_name(formatted_name)
            nodes.append({
                "position": i,
                "global_id": gid,
                "type": parsed["type"],
                "name": parsed["name"]
            })
            
        # Edges
        edges = []
        for i, (rel_id, rel_name) in enumerate(zip(path.relations, path.relation_names)):
            source_type = nodes[i]["type"]
            target_type = nodes[i+1]["type"]
            edges.append({
                "position": i,
                "relation_id": rel_id,
                "relation_name": rel_name,
                "source_type": source_type,
                "target_type": target_type
            })
            
        # Path object
        path_obj = {
            "path_id": rank - 1,
            "rank": rank,
            "length": len(path.relations),
            "scores": {
                "f_score": path.f_score,
                "g_score": path.g_score,
                "h_score": path.h_score,
                "embed_score": path.embed_score
            },
            "nodes": nodes,
            "edges": edges
        }
        paths_json.append(path_obj)

    final_json = {
        "query": {
            "user_id": query_user,
            "item_id": query_item
        },
        "metadata": metadata,
        "paths": paths_json
    }
    
    return final_json

def save_paths_json(json_data: Dict[str, Any], filepath: str) -> None:
    """Saves the JSON structure to a file."""
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)
