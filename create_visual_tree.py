#!/usr/bin/env python3
"""
Script to create a visual tree structure from the project_tree.json file.
Shows directories, files, classes, methods, functions, and imports.
"""

import json
import ast
import os
import io
import sys
from typing import Dict, List, Any, Set


def extract_imports_and_variables(file_path: str) -> Dict[str, List[str]]:
    """Extract imports and module-level variables from a Python file."""
    imports = []
    variables = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module:
                        imports.append(f"{module}.{alias.name}")
                    else:
                        imports.append(alias.name)
            elif isinstance(node, ast.Assign):
                # Check if this is a module-level assignment
                for parent in ast.walk(tree):
                    if isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
                        if node in ast.walk(parent):
                            break
                else:
                    # This is a module-level assignment
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append(target.id)
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return {"imports": imports, "variables": variables}


def print_tree_item(name: str, prefix: str, is_last: bool, item_type: str = ""):
    """Print a tree item with proper formatting."""
    connector = "└── " if is_last else "├── "
    type_label = f" ({item_type})" if item_type else ""
    print(f"{prefix}{connector}{name}{type_label}")


def get_child_prefix(prefix: str, is_last: bool) -> str:
    """Get the prefix for child items."""
    return prefix + ("    " if is_last else "│   ")


def print_python_details(file_info: Dict[str, Any], prefix: str, file_path: str):
    """Print detailed information about a Python file."""
    # Get additional info like imports and variables
    extra_info = extract_imports_and_variables(file_path)
    
    all_items = []
    
    # Add imports
    if extra_info["imports"]:
        all_items.append(("Imports", extra_info["imports"]))
    
    # Add variables
    if extra_info["variables"]:
        all_items.append(("Variables", extra_info["variables"]))
    
    # Add classes
    for class_info in file_info.get("classes", []):
        all_items.append(("Class", class_info))
    
    # Add functions
    for func_name in file_info.get("functions", []):
        all_items.append(("Function", func_name))
    
    # Print all items
    for i, (item_type, item_data) in enumerate(all_items):
        is_last_item = i == len(all_items) - 1
        
        if item_type == "Imports":
            imports_str = ", ".join(item_data)
            print_tree_item(f"Imports: {imports_str}", prefix, is_last_item)
        
        elif item_type == "Variables":
            variables_str = ", ".join(item_data)
            print_tree_item(f"Variables: {variables_str}", prefix, is_last_item)
        
        elif item_type == "Function":
            print_tree_item(f"Function: {item_data}()", prefix, is_last_item)
        
        elif item_type == "Class":
            class_name = item_data["name"]
            methods = item_data.get("methods", [])
            
            print_tree_item(f"Class: {class_name}", prefix, is_last_item)
            
            # Print methods
            if methods:
                method_prefix = get_child_prefix(prefix, is_last_item)
                for j, method in enumerate(methods):
                    is_last_method = j == len(methods) - 1
                    print_tree_item(f"{method}()", method_prefix, is_last_method)


def print_tree_recursive(tree_data: Dict[str, Any], prefix: str = "", is_last: bool = True):
    """Recursively print the tree structure."""
    
    if tree_data["type"] == "directory":
        # Don't print the root directory name if it's just "."
        if tree_data["name"] != ".":
            print_tree_item(tree_data["name"], prefix, is_last)
            prefix = get_child_prefix(prefix, is_last)
        
        children = tree_data.get("children", {})
        child_items = list(children.items())
        
        for i, (child_name, child_data) in enumerate(child_items):
            is_last_child = i == len(child_items) - 1
            
            if child_data["type"] == "directory":
                print_tree_recursive(child_data, prefix, is_last_child)
            else:
                # It's a file
                print_tree_item(child_name, prefix, is_last_child)
                
                # If it's a Python file, print additional details
                if child_name.endswith('.py'):
                    child_prefix = get_child_prefix(prefix, is_last_child)
                    print_python_details(child_data, child_prefix, child_data["path"])


def main():
    """Generate and print the visual tree structure."""
    
    # Load the project tree JSON
    try:
        with open('datasets/project_tree.json', 'r', encoding='utf-8') as f:
            project_tree = json.load(f)
    except FileNotFoundError:
        print("Error: datasets/project_tree.json not found. Please run the project tree generator first.")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return
    
    # Generate the tree structure as text
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        print("Project Tree Structure:")
        print("=" * 50)
        print()
        
        # Print the tree structure
        print_tree_recursive(project_tree)
        
        print()
        print("=" * 50)
        print("Tree generation complete!")
        
    finally:
        sys.stdout = original_stdout
    
    # Get the captured output
    tree_output = output_buffer.getvalue()
    
    # Print to console
    print(tree_output)
    
    # Save to file with proper encoding
    try:
        with open('datasets/project_tree_visual.txt', 'w', encoding='utf-8') as f:
            f.write(tree_output)
        print(f"Visual tree structure saved to: datasets/project_tree_visual.txt")
    except Exception as e:
        print(f"Error saving to file: {e}")


if __name__ == "__main__":
    main() 