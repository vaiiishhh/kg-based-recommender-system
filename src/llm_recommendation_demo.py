import argparse
import os
import json
from kg_path_extractor import KGPathExtractor, Config
from kg_path_to_json import paths_to_json, save_paths_json
from kg_entity_context import KGEntityContextManager
from llm_prompt_builder import LLMPromptBuilder

def main():
    parser = argparse.ArgumentParser(description="End-to-end demo for LLM Recommendation path extraction and prompt generation.")
    parser.add_argument("--data_dir", default="kg_data/", help="Path to kg_data/ folder")
    parser.add_argument("--user", required=True, help="User ID string")
    parser.add_argument("--item", required=True, help="Item ASIN string")
    parser.add_argument("--top_k", type=int, default=5, help="Number of paths to return")
    parser.add_argument("--output_dir", default="output/", help="Directory to save JSON and prompts")
    
    # Pathway configs
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=64)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Initialize Path Extractor
    print("[1/4] Initializing KG Path Extractor...")
    cfg = Config(
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        top_k_paths=args.top_k
    )
    extractor = KGPathExtractor(data_dir=args.data_dir, config=cfg)
    
    # 2. Extract paths
    print(f"[2/4] Extracting top {args.top_k} paths for User {args.user} -> Item {args.item}...")
    try:
        results = extractor.find_paths(args.user, args.item, top_k=args.top_k)
    except KeyError as e:
        print(f"Error: {e}")
        return
    
    if not results:
        print("Warning: No paths found!")
        
    # 3. Convert to JSON
    print("[3/4] Converting paths to standardized JSON...")
    paths_json = paths_to_json(args.user, args.item, results, extractor, cfg)
    
    json_path = os.path.join(args.output_dir, f"{args.user}_{args.item}_paths.json")
    save_paths_json(paths_json, json_path)
    print(f"      Saved JSON to {json_path}")
    
    # 4. Build Prompt
    print("[4/4] Building Deterministic LLM Prompt...")
    context_manager = KGEntityContextManager(data_dir=args.data_dir)
    prompt_builder = LLMPromptBuilder(context_manager)
    
    prompt = prompt_builder.build_prompt(paths_json)
    
    prompt_path = os.path.join(args.output_dir, f"{args.user}_{args.item}_prompt.txt")
    prompt_str = f"=== SYSTEM PROMPT ===\n{prompt['system']}\n\n=== USER PROMPT ===\n{prompt['user']}"
    with open(prompt_path, 'w') as f:
        f.write(prompt_str)
    print(f"      Saved Prompt to {prompt_path}")
    
    print("\n--- Pipeline Complete! ---")
    print("\nPROMPT PREVIEW:")
    lines = prompt_str.splitlines()
    print("\n".join(lines[:20] + ["..."] + lines[-10:]))

if __name__ == "__main__":
    main()
