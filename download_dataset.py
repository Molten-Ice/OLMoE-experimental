import os
import json
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi

api = huggingface_hub.HfApi()
with open('../huggingface_apikey.json', 'r') as f:
    api_key = json.load(f)['api_key']

huggingface_hub.login(token=api_key)

os.makedirs('data', exist_ok=True)

def get_filepath_to_size():
    api = HfApi()
    repo_info = api.repo_info(
        repo_id="allenai/OLMoE-mix-0924",
        repo_type="dataset",
        files_metadata=True
    )
    filepath_to_size = {sibling.rfilename: sibling.size for sibling in repo_info.siblings}
    with open('data/filepath_to_size.json', 'w') as f:
        json.dump(filepath_to_size, f, indent=2)
    return filepath_to_size


def build_directory_tree(filepath_to_size):
    """
    Build a nested directory structure from a dict {full_path: size}.

    Additional rules:
      a) If files are "similar" after removing trailing digits/hyphens, put them in a parent folder named after that base.
         If the base name is empty after reduction, use 'folder'.
      b) If there is only one item in that group, do not make a subfolder, keep it in the parent.

    Returns a nested dict with structure:
      {
        "somefile.txt": {
          "type": "file",
          "path": "somefile.txt",
          "size": 1234,
          "base_name": "somefile"
        },
        "somename": {
          "type": "directory",
          "somefile1.json": {...},
          "somefile2.json": {...}
        },
        ...
      }
    """

    # -------------------------------------------------------------------------
    # 1. Build a raw tree of directories (without grouping by base_name yet).
    # -------------------------------------------------------------------------
    raw_tree = {}

    for path_str, size in filepath_to_size.items():
        parts = Path(path_str).parts
        node = raw_tree

        # Traverse or create subdirectories
        for part in parts[:-1]:
            if part not in node:
                # Each directory node can stash subdirectories under normal dict keys,
                # so initialize them as dict. We'll mark them "type=directory" later.
                node[part] = {}
            node = node[part]

        filename = parts[-1]
        # Store just a placeholder for the file
        # We'll handle base_name and so forth in a "finalize" step.
        node[filename] = {
            "path": path_str,
            "size": size,
        }

    # -------------------------------------------------------------------------
    # Utility to get the "base name" after trimming digits/hyphens.
    # -------------------------------------------------------------------------
    def get_base_name(file_path):
        """
        Extracts a 'base name' from a file path by removing multiple known suffixes
        (like .gz, .zst, .xz, .bz2, .jsonl, .json, etc.). Then checks if the remainder
        is purely numeric. If so, returns 'numeric' to group them all together.
        Otherwise removes trailing digits/hyphens. Returns 'folder' if nothing remains.
        """
        import re
        p = Path(file_path)
        
        # Start with the file name (including extensions)
        name = p.name
        
        # List of known suffixes to remove:
        known_suffixes = [".gz", ".zst", ".xz", ".bz2", ".jsonl", ".json"]
        
        # Keep stripping suffixes as long as they match
        changed = True
        while changed:
            changed = False
            for sfx in known_suffixes:
                if name.endswith(sfx):
                    name = name[: -len(sfx)]
                    changed = True
        
        # Now check if purely numeric:
        if name.isdigit():
            return 'numeric'  # group all numeric files together
        
        # If it's all digits/hyphens, call it 'folder'
        if all(c.isdigit() or c == '-' for c in name):
            return 'folder'
        
        # Otherwise, strip trailing digits/hyphens from the end
        name = re.sub(r'[\d-]+$', '', name)
        name = name.strip()
        
        return name if name else "folder"

    # -------------------------------------------------------------------------
    # 2. Transform the raw_tree into our final structure:
    #    - Mark directories
    #    - Group files by base_name
    #    - If there's only one file per base_name, do not create a subdirectory
    # -------------------------------------------------------------------------
    def finalize_tree(node):
        """
        Recursively transform node into the final { name -> {...}} structure,
        grouping files by base_name, creating subdirectories only where needed.
        """
        # separate subdicts from final-level files in this node
        child_keys = list(node.keys())
        subdirectories = []
        files_here = []

        for key in child_keys:
            # If node[key] is a dict but doesn't have "path", it must be a directory
            if isinstance(node[key], dict) and "path" not in node[key]:
                subdirectories.append(key)
            else:
                files_here.append(key)

        # Recursively finalize subdirectories first
        for dname in subdirectories:
            sub_node = finalize_tree(node[dname])
            # Mark them as a directory
            sub_node["type"] = "directory"
            node[dname] = sub_node

        # Group files by base_name
        from collections import defaultdict
        base_to_files = defaultdict(list)

        for fname in files_here:
            file_info = node[fname]
            path_str = file_info["path"]
            base = get_base_name(path_str)

            # We'll fill in more metadata in "file_info"
            file_info["type"] = "file"
            file_info["base_name"] = base
            base_to_files[base].append((fname, file_info))

        # Now either go directly in node, or group them under a subdirectory
        # for each base
        for base, items in base_to_files.items():
            if len(items) == 1:
                # If there's only one file in this base group, we do not create a subfolder
                fname, file_info = items[0]
                node[fname] = file_info
            else:
                # More than 1 file => create subdirectory for that base
                node[base] = {"type": "directory"}
                for fname, file_info in items:
                    node[base][fname] = file_info
                    # Remove the file from its original location
                    if fname in node:
                        del node[fname]

        return node

    # Finalize from the top-level
    final_structure = finalize_tree(raw_tree)

    # Mark the root as a directory (label the top node). 
    # Usually not too critical, but for consistency:
    final_structure["type"] = "directory"

    return final_structure

def print_tree(node, prefix="", is_last=True, name="", is_root=True):
    """Build a string representation of the tree structure, limiting to 3 files per directory."""
    result = []
    
    if not is_root:
        connector = "└── " if is_last else "├── "
        
        if isinstance(node, dict) and 'path' in node:
            size = node.get('size', 0)
            size_str = f" ({size / 1_000_000_000:.2f} GB)" if size > 100_000_000 else f" ({size / 1_000_000:.1f} MB)"
            result.append(prefix + connector + Path(node['path']).name + size_str)
        else:
            total_size = calculate_dir_size(node)
            size_str = f" ({total_size / 1_000_000_000:.2f} GB)" if total_size > 100_000_000 else f" ({total_size / 1_000_000:.1f} MB)"
            display_name = name if name else "[Directory]"
            result.append(prefix + connector + display_name + size_str)
    else:
        total_size = calculate_dir_size(tree_structure['data'])
        size_str = f" ({total_size / 1_000_000_000:.2f} GB)" if total_size > 100_000_000 else f" ({total_size / 1_000_000:.1f} MB)"
        result.append(f"data{size_str}")

    if isinstance(node, dict):
        children = {k: v for k, v in node.items() if k not in ['type', 'path', 'size', 'base_name']}
        items = list(children.items())
        
        is_leaf_dir = all(isinstance(v, dict) and 'path' in v for v in children.values())
        
        display_items = items[:3] if is_leaf_dir else items
        hidden_items = items[3:] if is_leaf_dir and len(items) > 3 else []
        hidden_count = len(hidden_items)
        
        for i, (child_name, child) in enumerate(display_items):
            new_prefix = prefix if is_root else (prefix + ("    " if is_last else "│   "))
            is_last_item = (i == len(display_items) - 1) and (hidden_count == 0)
            result.extend(print_tree(child, new_prefix, is_last_item, child_name, is_root=False))
        
        if hidden_count > 0:
            new_prefix = prefix if is_root else (prefix + ("    " if is_last else "│   "))
            hidden_size = sum(child['size'] for _, child in hidden_items if isinstance(child, dict) and 'size' in child)
            if hidden_size <= 100_000_000:  # 100 million
                size_str = f"{hidden_size / 1_000_000:.1f}MB"
            else:
                size_str = f"{hidden_size / 1_000_000_000:.1f}GB"
            result.append(f"{new_prefix}└── ... ({hidden_count} hidden files, {size_str} total)")

    return result


def calculate_dir_size(node):
    """Calculate total size of a directory and all its contents"""
    if isinstance(node, dict):
        if 'path' in node:  # It's a file
            return node.get('size', 0)
        else:  # It's a directory
            return sum(calculate_dir_size(child) for child in node.values() if isinstance(child, dict))
    return 0

## Visualize the dataset structure ##

# filepath_to_size = get_filepath_to_size()
with open('data/filepath_to_size.json', 'r') as f:
    filepath_to_size = json.load(f)

# filepath_to_size = {k: v for k, v in filepath_to_size.items() if 'open-web-math' in k}

print('Building tree...')
tree_structure = build_directory_tree(filepath_to_size)
print('Tree built.')

output_path = 'data/dataset_structure.json'
with open(output_path, 'w') as f:
    json.dump(tree_structure, f, indent=2)


tree_output = '\n'.join(print_tree(tree_structure))
with open('data/dataset_structure.txt', 'w') as f:
    f.write(tree_output)

print('-'*100)
print(tree_output)
print('-'*100)

## Download the dataset ##
# Define the base path for downloads
os.makedirs('data', exist_ok=True)
dataset_path = "data/datasets"
os.makedirs(dataset_path, exist_ok=True)

# Download files dclm-0000 through dclm-0004
for i in range(5):
    file_name = f"dclm-{i:04d}.json.zst"
    local_path = os.path.join(dataset_path, file_name)

    if os.path.exists(local_path):
        print(f"Skipping {file_name} because it already exists.")
        continue
    
    print(f"Downloading {file_name}...")
    huggingface_hub.hf_hub_download(
        repo_id="allenai/OLMoE-mix-0924",
        repo_type="dataset",
        filename=f"data/dclm/{file_name}", # remote path
        local_dir=dataset_path
    )