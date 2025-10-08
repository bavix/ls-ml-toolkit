#!/usr/bin/env python3
"""
Minimal CLI UI components for Label Studio ML Toolkit
Only includes components that are actually used
"""

import sys
import time
from typing import List
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    # Basic colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Reset
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for non-terminal output"""
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                setattr(cls, attr, '')

class Icons:
    """Unicode icons for beautiful output"""
    SUCCESS = '✅'
    ERROR = '❌'
    WARNING = '⚠️'
    INFO = 'ℹ️'
    LOADING = '⏳'
    CONFIG = '⚙️'
    OPTIMIZE = '🔧'
    FOLDER = '📁'
    FILE = '📄'
    
    # Table symbols
    DASH = '─'
    VERTICAL = '│'
    CORNER_TL = '┌'
    CORNER_TR = '┐'
    CORNER_BL = '└'
    CORNER_BR = '┘'
    T_LEFT = '├'
    T_RIGHT = '┤'
    T_UP = '┬'
    T_DOWN = '┴'
    CROSS_PIECE = '┼'

class Banner:
    """Beautiful banner for application startup"""
    
    @staticmethod
    def display(version=None):
        """Display the application banner"""
        if version is None:
            try:
                from . import __version__
                version = __version__
            except ImportError:
                version = "1.0.2"
        
        banner = f"""
{Colors.BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.BRIGHT_CYAN}║{Colors.RESET}                    {Colors.BOLD}{Colors.BRIGHT_WHITE}LS-ML-Toolkit{Colors.RESET}                    {Colors.BRIGHT_CYAN}║{Colors.RESET}
{Colors.BRIGHT_CYAN}║{Colors.RESET}              {Colors.DIM}Label Studio → YOLO → ONNX{Colors.RESET}              {Colors.BRIGHT_CYAN}║{Colors.RESET}
{Colors.BRIGHT_CYAN}║{Colors.RESET}                    {Colors.DIM}v{version}{Colors.RESET}                    {Colors.BRIGHT_CYAN}║{Colors.RESET}
{Colors.BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
        print(banner)

class Table:
    """Beautiful table for displaying structured data"""
    
    def __init__(self, headers: List[str], style: str = "modern"):
        self.headers = headers
        self.rows = []
        self.style = style
        
    def add_row(self, row: List[str]):
        """Add a row to the table"""
        self.rows.append(row)
        
    def display(self):
        """Display the table"""
        if not self.rows:
            return
            
        # Calculate column widths
        col_widths = [len(header) for header in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Display header
        self._print_separator(col_widths, "top")
        self._print_row(self.headers, col_widths, is_header=True)
        self._print_separator(col_widths, "middle")
        
        # Display rows
        for row in self.rows:
            self._print_row(row, col_widths)
            
        self._print_separator(col_widths, "bottom")
        
    def _print_separator(self, col_widths: List[int], position: str):
        """Print table separator"""
        if self.style == "modern":
            if position == "top":
                left = Icons.CORNER_TL
                middle = Icons.T_DOWN
                right = Icons.CORNER_TR
            elif position == "middle":
                left = Icons.T_LEFT
                middle = Icons.CROSS_PIECE
                right = Icons.T_RIGHT
            else:  # bottom
                left = Icons.CORNER_BL
                middle = Icons.T_UP
                right = Icons.CORNER_BR
                
            line = left
            for i, width in enumerate(col_widths):
                line += Icons.DASH * (width + 2)
                if i < len(col_widths) - 1:
                    line += middle
            line += right
            
            print(f"{Colors.DIM}{line}{Colors.RESET}")
            
    def _print_row(self, row: List[str], col_widths: List[int], is_header: bool = False):
        """Print a table row"""
        line = f"{Icons.VERTICAL} "
        
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            if is_header:
                cell_str = f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{str(cell):<{width}}{Colors.RESET}"
            else:
                cell_str = f"{str(cell):<{width}}"
            line += cell_str
            if i < len(row) - 1:
                line += f" {Icons.VERTICAL} "
            else:
                line += f" {Icons.VERTICAL}"
                
        print(f"{Colors.DIM}{line}{Colors.RESET}")

class StatusDisplay:
    """Beautiful status display for long-running operations"""
    
    def __init__(self, title: str):
        self.title = title
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 0
        self.steps = []
        
    def add_step(self, step: str):
        """Add a step to the process"""
        self.steps.append(step)
        self.total_steps = len(self.steps)
        
    def start(self):
        """Start the process display"""
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{Icons.CONFIG} {self.title}{Colors.RESET}")
        print(f"{Colors.DIM}{Icons.DASH * 60}{Colors.RESET}")
        
    def update_step(self, step: int, message: str = ""):
        """Update current step"""
        self.current_step = step
        if step < len(self.steps):
            step_name = self.steps[step]
            status = f"{Colors.BRIGHT_GREEN}{Icons.SUCCESS}{Colors.RESET}" if step < self.current_step else f"{Colors.BRIGHT_YELLOW}{Icons.LOADING}{Colors.RESET}"
            print(f"  {status} {step_name}")
            if message:
                print(f"     {Colors.DIM}{message}{Colors.RESET}")
                
    def complete(self):
        """Mark process as complete"""
        elapsed = time.time() - self.start_time
        print(f"\n{Colors.BRIGHT_GREEN}{Icons.SUCCESS} Process completed in {elapsed:.1f}s{Colors.RESET}")

class FileTree:
    """Beautiful file tree display"""
    
    @staticmethod
    def display(path: Path, max_depth: int = 3, show_hidden: bool = False):
        """Display a file tree"""
        def _print_tree(p: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
                
            if not p.exists():
                return
                
            # Get children
            try:
                children = sorted([child for child in p.iterdir() 
                                 if not child.name.startswith('.') or show_hidden])
            except PermissionError:
                return
                
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{child.name}")
                
                if child.is_dir() and depth < max_depth:
                    extension = "    " if is_last else "│   "
                    _print_tree(child, prefix + extension, depth + 1)
                    
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{Icons.FOLDER} {path}{Colors.RESET}")
        _print_tree(path)

# Disable colors if not in terminal
if not sys.stdout.isatty():
    Colors.disable()