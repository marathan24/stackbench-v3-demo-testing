"""
Interactive codebase map and dependency visualization.

Provides an interactive tree view of the codebase with annotations
including module purpose, LOC, dependencies, and test coverage hints.
"""

import ast
import json
from pathlib import Path


class CodeMapGenerator:
    """Generate interactive code map with annotations."""

    def __init__(self, project_root: Path):
        """Initialize code map generator.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.stackbench_root = self.project_root / "stackbench"
        self.frontend_root = self.project_root / "frontend"

    def count_lines(self, file_path: Path) -> int:
        """Count lines of code in a file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                return sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
        except Exception:
            return 0

    def extract_imports(self, file_path: Path) -> list[str]:
        """Extract import statements from a Python file."""
        imports = []
        try:
            with open(file_path, encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'stackbench' in alias.name:
                            imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and 'stackbench' in node.module:
                        imports.append(node.module)
        except Exception:
            pass

        return imports

    def get_module_purpose(self, file_path: Path) -> str:
        """Extract module docstring as purpose."""
        try:
            with open(file_path, encoding='utf-8') as f:
                tree = ast.parse(f.read())
                docstring = ast.get_docstring(tree)
                if docstring:
                    # Get first line only
                    return docstring.split('\n')[0]
        except Exception:
            pass
        return "No description available"

    def has_tests(self, module_path: Path) -> bool:
        """Check if module has corresponding test file."""
        test_path = self.project_root / "tests" / f"test_{module_path.name}"
        return test_path.exists()

    def generate_python_tree(self) -> dict:
        """Generate annotated tree for Python package."""
        tree = {
            "name": "stackbench (Python Package)",
            "type": "package",
            "loc": 0,
            "children": []
        }

        # Walk through stackbench directory
        for item in sorted(self.stackbench_root.iterdir()):
            if item.name.startswith('.') or item.name == '__pycache__':
                continue

            if item.is_dir():
                # It's a subpackage
                subpackage = self._process_directory(item)
                tree["children"].append(subpackage)
                tree["loc"] += subpackage.get("loc", 0)
            elif item.suffix == '.py':
                # It's a module
                module = self._process_python_file(item)
                tree["children"].append(module)
                tree["loc"] += module.get("loc", 0)

        return tree

    def _process_directory(self, dir_path: Path) -> dict:
        """Process a Python package directory."""
        node = {
            "name": dir_path.name,
            "type": "subpackage",
            "path": str(dir_path.relative_to(self.project_root)),
            "loc": 0,
            "children": []
        }

        # Add README if exists
        readme = dir_path / "README.md"
        if readme.exists():
            node["has_readme"] = True
            node["readme_path"] = str(readme.relative_to(self.project_root))

        # Process all Python files in this directory
        for item in sorted(dir_path.iterdir()):
            if item.name.startswith('.') or item.name == '__pycache__':
                continue

            if item.suffix == '.py':
                module = self._process_python_file(item)
                node["children"].append(module)
                node["loc"] += module.get("loc", 0)

        return node

    def _process_python_file(self, file_path: Path) -> dict:
        """Process a single Python file."""
        loc = self.count_lines(file_path)
        imports = self.extract_imports(file_path)
        purpose = self.get_module_purpose(file_path)
        has_tests = self.has_tests(file_path)

        return {
            "name": file_path.name,
            "type": "module",
            "path": str(file_path.relative_to(self.project_root)),
            "loc": loc,
            "purpose": purpose,
            "imports": imports,
            "has_tests": has_tests
        }

    def generate_frontend_tree(self) -> dict:
        """Generate annotated tree for frontend."""
        if not self.frontend_root.exists():
            return None

        tree = {
            "name": "frontend (React + TypeScript)",
            "type": "package",
            "loc": 0,
            "children": []
        }

        src_dir = self.frontend_root / "src"
        if src_dir.exists():
            for item in sorted(src_dir.iterdir()):
                if item.name.startswith('.'):
                    continue

                if item.is_dir():
                    subdir = self._process_frontend_directory(item)
                    tree["children"].append(subdir)
                    tree["loc"] += subdir.get("loc", 0)
                elif item.suffix in ['.ts', '.tsx']:
                    module = self._process_frontend_file(item)
                    tree["children"].append(module)
                    tree["loc"] += module.get("loc", 0)

        return tree

    def _process_frontend_directory(self, dir_path: Path) -> dict:
        """Process a frontend directory."""
        node = {
            "name": dir_path.name,
            "type": "directory",
            "path": str(dir_path.relative_to(self.project_root)),
            "loc": 0,
            "children": []
        }

        for item in sorted(dir_path.iterdir()):
            if item.name.startswith('.'):
                continue

            if item.suffix in ['.ts', '.tsx']:
                module = self._process_frontend_file(item)
                node["children"].append(module)
                node["loc"] += module.get("loc", 0)

        return node

    def _process_frontend_file(self, file_path: Path) -> dict:
        """Process a single frontend file."""
        loc = self.count_lines(file_path)

        # Determine type
        if file_path.suffix == '.tsx':
            file_type = "component" if file_path.name[0].isupper() else "module"
        else:
            file_type = "module"

        return {
            "name": file_path.name,
            "type": file_type,
            "path": str(file_path.relative_to(self.project_root)),
            "loc": loc
        }

    def print_tree(self, tree: dict, indent: int = 0, show_details: bool = True):
        """Print tree in a readable format."""
        from rich.console import Console
        from rich.tree import Tree as RichTree

        console = Console()

        def build_rich_tree(node: dict, parent=None):
            """Recursively build Rich tree."""
            if node["type"] == "package" or node["type"] == "subpackage":
                label = f"[bold cyan]{node['name']}[/bold cyan] ([yellow]{node['loc']}[/yellow] LOC)"
            elif node["type"] == "module":
                test_indicator = "✓" if node.get("has_tests") else "✗"
                label = f"[green]{node['name']}[/green] ([yellow]{node['loc']}[/yellow] LOC, tests: {test_indicator})"
                if show_details and node.get("purpose"):
                    label += f"\n  [dim]{node['purpose']}[/dim]"
                if show_details and node.get("imports"):
                    imports_str = ", ".join(set(node["imports"][:3]))  # Show first 3 unique
                    if len(node["imports"]) > 3:
                        imports_str += f" +{len(node['imports']) - 3} more"
                    label += f"\n  [dim]Imports: {imports_str}[/dim]"
            elif node["type"] == "component":
                label = f"[magenta]{node['name']}[/magenta] ([yellow]{node['loc']}[/yellow] LOC)"
            elif node["type"] == "directory":
                label = f"[cyan]{node['name']}[/cyan] ([yellow]{node['loc']}[/yellow] LOC)"
            else:
                label = f"{node['name']} ([yellow]{node['loc']}[/yellow] LOC)"

            if parent is None:
                tree_node = RichTree(label)
            else:
                tree_node = parent.add(label)

            for child in node.get("children", []):
                build_rich_tree(child, tree_node)

            return tree_node

        rich_tree = build_rich_tree(tree)
        console.print(rich_tree)

    def export_json(self, output_path: Path):
        """Export code map to JSON file."""
        python_tree = self.generate_python_tree()
        frontend_tree = self.generate_frontend_tree()

        data = {
            "project_root": str(self.project_root),
            "python": python_tree,
            "frontend": frontend_tree,
            "total_python_loc": python_tree["loc"],
            "total_frontend_loc": frontend_tree["loc"] if frontend_tree else 0
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return data


def generate_dependency_graph_python(output_path: Path):
    """Generate dependency graph for Python modules using Graphviz."""
    try:
        import graphviz
    except ImportError:
        print("Warning: graphviz not installed. Install with: pip install graphviz")
        return None

    dot = graphviz.Digraph(comment='Stackbench Module Dependencies')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    # Define module groups for better visualization
    groups = {
        'cli': ['cli'],
        'pipeline': ['pipeline.runner'],
        'agents': ['agents.extraction_agent', 'agents.api_signature_validation_agent',
                  'agents.code_example_validation_agent', 'agents.clarity_agent'],
        'walkthroughs': ['walkthroughs.walkthrough_generate_agent',
                        'walkthroughs.walkthrough_audit_agent',
                        'walkthroughs.mcp_server'],
        'infrastructure': ['cache.manager', 'repository.manager', 'hooks.manager'],
        'schemas': ['schemas']
    }

    # Add nodes
    for group_name, modules in groups.items():
        with dot.subgraph(name=f'cluster_{group_name}') as c:
            c.attr(label=group_name.capitalize(), style='filled', color='lightgrey')
            for module in modules:
                node_id = module.replace('.', '_')
                display_name = module.split('.')[-1]
                c.node(node_id, display_name)

    # Add edges (dependencies)
    dependencies = [
        ('cli', 'pipeline_runner'),
        ('cli', 'walkthroughs_walkthrough_generate_agent'),
        ('cli', 'walkthroughs_walkthrough_audit_agent'),
        ('cli', 'repository_manager'),
        ('pipeline_runner', 'repository_manager'),
        ('pipeline_runner', 'cache_manager'),
        ('pipeline_runner', 'agents_extraction_agent'),
        ('pipeline_runner', 'agents_api_signature_validation_agent'),
        ('pipeline_runner', 'agents_code_example_validation_agent'),
        ('pipeline_runner', 'agents_clarity_agent'),
        ('agents_extraction_agent', 'hooks_manager'),
        ('agents_api_signature_validation_agent', 'hooks_manager'),
        ('agents_code_example_validation_agent', 'hooks_manager'),
        ('agents_clarity_agent', 'hooks_manager'),
        ('walkthroughs_walkthrough_generate_agent', 'hooks_manager'),
        ('walkthroughs_walkthrough_audit_agent', 'walkthroughs_mcp_server'),
        ('hooks_manager', 'schemas'),
    ]

    for source, target in dependencies:
        dot.edge(source, target)

    # Render to file
    output_file = str(output_path.with_suffix(''))
    dot.render(output_file, format='png', cleanup=True)
    dot.render(output_file, format='svg', cleanup=True)

    print("Generated dependency graphs:")
    print(f"  - {output_file}.png")
    print(f"  - {output_file}.svg")

    return dot


def generate_dependency_graph_frontend(output_path: Path):
    """Generate dependency graph for frontend components."""
    try:
        import graphviz
    except ImportError:
        print("Warning: graphviz not installed. Install with: pip install graphviz")
        return None

    dot = graphviz.Digraph(comment='Frontend Component Dependencies')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgreen')

    # Define components
    components = [
        'App.tsx',
        'RunSelector.tsx',
        'RunInfo.tsx',
        'Tabs.tsx',
        'CodeBlockWithValidation.tsx',
        'WalkthroughViewer.tsx',
        'GapCard.tsx',
        'MarkdownViewer.tsx',
        'Settings.tsx'
    ]

    services = [
        'api.ts'
    ]

    # Add nodes
    with dot.subgraph(name='cluster_components') as c:
        c.attr(label='Components', style='filled', color='lightgrey')
        for comp in components:
            c.node(comp.replace('.', '_'), comp)

    with dot.subgraph(name='cluster_services') as c:
        c.attr(label='Services', style='filled', color='lightblue')
        for svc in services:
            c.node(svc.replace('.', '_'), svc)

    # Add edges (component hierarchy)
    dependencies = [
        ('App_tsx', 'RunSelector_tsx'),
        ('App_tsx', 'RunInfo_tsx'),
        ('App_tsx', 'Tabs_tsx'),
        ('App_tsx', 'WalkthroughViewer_tsx'),
        ('App_tsx', 'Settings_tsx'),
        ('Tabs_tsx', 'CodeBlockWithValidation_tsx'),
        ('Tabs_tsx', 'MarkdownViewer_tsx'),
        ('WalkthroughViewer_tsx', 'GapCard_tsx'),
        ('App_tsx', 'api_ts'),
        ('RunSelector_tsx', 'api_ts'),
        ('CodeBlockWithValidation_tsx', 'api_ts'),
        ('WalkthroughViewer_tsx', 'api_ts'),
    ]

    for source, target in dependencies:
        dot.edge(source, target)

    # Render to file
    output_file = str(output_path.with_suffix(''))
    dot.render(output_file, format='png', cleanup=True)
    dot.render(output_file, format='svg', cleanup=True)

    print("Generated frontend dependency graphs:")
    print(f"  - {output_file}.png")
    print(f"  - {output_file}.svg")

    return dot
