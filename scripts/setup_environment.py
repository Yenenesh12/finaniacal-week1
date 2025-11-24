
"""
Script to set up the development environment.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"✓ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running '{command}': {e}")
        return False

def setup_environment():
    """Set up the development environment."""
    print("Setting up development environment...")
    
    # Create virtual environment
    if not run_command("python -m venv venv"):
        print("Failed to create virtual environment")
        return False
    
    # Determine activation command based on OS
    if sys.platform == "win32":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{pip_path} install --upgrade pip"):
        print("Failed to upgrade pip")
        return False
    
    # Install requirements
    if not run_command(f"{pip_path} install -r requirements.txt"):
        print("Failed to install requirements")
        return False
    
    print("\n✅ Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run tests: pytest tests/")
    print("3. Start Jupyter: jupyter notebook")
    
    return True

if __name__ == "__main__":
    setup_environment()