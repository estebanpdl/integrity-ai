#!/usr/bin/env python3
"""
Script to generate a project tree showing directory structure and Python file analysis.
Excludes files/directories listed in .gitignore.
For each .py file, extracts methods and classes.
"""

import os
import json
import ast
import fnmatch
from pathlib import Path
from typing import Dict, List, Any, Set


def parse_gitignore(gitignore_path: str) -> Set[str]:
    """Parse .gitignore file and return set of patterns to ignore."""
    ignore_patterns = set()
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove trailing slashes for directories
                    pattern = line.rstrip('/')
                    ignore_patterns.add(pattern)
                    # Also add the directory pattern with trailing slash
                    if not pattern.endswith('/'):
                        ignore_patterns.add(pattern + '/')
    
    return ignore_patterns


def should_ignore(path: str, ignore_patterns: Set[str]) -> bool:
    """Check if a path should be ignored based on gitignore patterns."""
    # Convert to forward slashes for consistent matching
    path = path.replace('\\', '/')
    
    for pattern in ignore_patterns:
        # Handle directory patterns
        if pattern.endswith('/'):
            if path.startswith(pattern) or ('/' + pattern in path):
                return True
        else:
            # Check if the pattern matches the full path or just the filename
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
            # Also check if any part of the path matches
            if ('/' + pattern + '/') in ('/' + path + '/'):
                return True
    
    return False


def extract_python_info(file_path: str) -> Dict[str, Any]:
    """Extract classes and methods from a Python file."""
    info = {
        'classes': [],
        'functions': [],
        'methods': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'methods': []
                }
                
                # Get methods within this class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info['methods'].append(item.name)
                
                info['classes'].append(class_info)
            
            elif isinstance(node, ast.FunctionDef):
                # Check if this function is at module level (not inside a class)
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if node in ast.walk(parent):
                            break
                else:
                    info['functions'].append(node.name)
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return info


def build_project_tree(root_path: str, ignore_patterns: Set[str]) -> Dict[str, Any]:
    """Build a project tree dictionary."""
    tree = {
        'type': 'directory',
        'name': os.path.basename(root_path) or 'root',
        'path': root_path,
        'children': {}
    }
    
    try:
        for item in os.listdir(root_path):
            item_path = os.path.join(root_path, item)
            relative_path = os.path.relpath(item_path, start=os.getcwd())
            
            # Skip if should be ignored
            if should_ignore(relative_path, ignore_patterns):
                continue
            
            if os.path.isdir(item_path):
                # Recursively build subtree for directories
                subtree = build_project_tree(item_path, ignore_patterns)
                if subtree['children']:  # Only add if directory has non-ignored contents
                    tree['children'][item] = subtree
            else:
                # Add file to tree
                file_info = {
                    'type': 'file',
                    'name': item,
                    'path': relative_path
                }
                
                # If it's a Python file, extract additional info
                if item.endswith('.py'):
                    python_info = extract_python_info(item_path)
                    file_info.update(python_info)
                
                tree['children'][item] = file_info
    
    except PermissionError:
        print(f"Permission denied: {root_path}")
    except Exception as e:
        print(f"Error processing {root_path}: {e}")
    
    return tree


def main():
    """Generate project tree and save to JSON."""
    # Parse .gitignore
    ignore_patterns = parse_gitignore('.gitignore')
    
    # Add some default patterns to ignore
    default_ignores = {'.git', '__pycache__', '*.pyc', '*.pyo', '*.pyd', '.DS_Store'}
    ignore_patterns.update(default_ignores)
    
    print("Generating project tree...")
    print(f"Ignoring patterns: {sorted(ignore_patterns)}")
    
    # Build the project tree
    project_tree = build_project_tree('.', ignore_patterns)
    
    # Save to datasets directory
    os.makedirs('datasets', exist_ok=True)
    output_path = os.path.join('datasets', 'project_tree.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project_tree, f, indent=2, ensure_ascii=False)
    
    print(f"Project tree saved to: {output_path}")
    
    # Print some statistics
    def count_items(tree_node):
        files = 0
        dirs = 0
        py_files = 0
        
        if tree_node['type'] == 'file':
            files += 1
            if tree_node['name'].endswith('.py'):
                py_files += 1
        else:
            dirs += 1
            for child in tree_node['children'].values():
                f, d, p = count_items(child)
                files += f
                dirs += d
                py_files += p
        
        return files, dirs, py_files
    
    total_files, total_dirs, total_py_files = count_items(project_tree)
    print(f"\nStatistics:")
    print(f"  Total directories: {total_dirs}")
    print(f"  Total files: {total_files}")
    print(f"  Python files: {total_py_files}")


if __name__ == "__main__":
    main() 