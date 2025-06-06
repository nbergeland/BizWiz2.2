# === INSTALL.PY ===
# Save this as: install.py

"""
Enhanced BizWizV2 Installation and Setup Script
Automatically sets up the environment and dependencies
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ Error in {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception in {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please upgrade to Python 3.8 or higher")
        return False

def create_env_template():
    """Create .env template file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        env_template = """# Enhanced BizWizV2 API Configuration
# Replace 'YOURAPIHERE' with your actual API keys

# Google Maps API Key (required)
# Get it from: https://console.cloud.google.com/apis/credentials
GOOGLE_API_KEY=YOURAPIHERE

# US Census Bureau API Key (required)
# Get it from: https://api.census.gov/data/key_signup.html
CENSUS_API_KEY=YOURAPIHERE

# RentCast API Key (required for rental data)
# Get it from: https://app.rentcast.io/app/api-access
RENTCAST_API_KEY=YOURAPIHERE
"""
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("✅ Created .env template file")
        print("📝 Please edit .env file and add your API keys")
        return True
    else:
        print("✅ .env file already exists")
        return True

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "cache_grand_forks_nd",
        "cache_fargo_nd", 
        "cache_bismarck_nd",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created directory structure")
    return True

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        print("Creating basic requirements.txt...")
        
        basic_requirements = """pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
requests>=2.28.0
googlemaps>=4.7.0
python-dotenv>=0.19.0
plotly>=5.10.0
dash>=2.6.0
dash-bootstrap-components>=1.2.0
PyYAML>=6.0
"""
        with open(requirements_file, 'w') as f:
            f.write(basic_requirements)
    
    # Install requirements
    if run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages"):
        return True
    else:
        print("Trying alternative installation method...")
        return run_command(f"{sys.executable} -m pip install --user -r requirements.txt", "Installing Python packages (user mode)")

def validate_installation():
    """Validate that key packages can be imported"""
    test_imports = [
        "pandas", "numpy", "sklearn", "requests", 
        "googlemaps", "plotly", "dash", "yaml"
    ]
    
    failed_imports = []
    for package in test_imports:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            failed_imports.append(package)
            print(f"❌ Failed to import {package}")
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("You may need to install these packages manually:")
        for package in failed_imports:
            print(f"  pip install {package}")
        return False
    else:
        print("✅ All packages imported successfully")
        return True

def create_quick_start_script():
    """Create a quick start script"""
    script_content = '''#!/usr/bin/env python3
"""
Enhanced BizWizV2 Quick Start Script
This script provides an easy way to run the enhanced analysis system.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file is properly configured"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please run: python install.py")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    if "YOURAPIHERE" in content:
        print("❌ .env file contains placeholder values!")
        print("Please edit .env file and add your actual API keys")
        return False
    
    return True

def main():
    print("🍗 Enhanced BizWizV2 - Quick Start")
    print("=" * 50)
    
    if not check_env_file():
        sys.exit(1)
    
    print("Select an option:")
    print("1. 🔄 Collect data for current city")
    print("2. 🔄 Collect data for specific city")
    print("3. 🌐 Start visualization app")
    print("4. 📊 List available cities")
    print("5. ⚙️  Add new city configuration")
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        subprocess.run([sys.executable, "enhanced_data_collection.py"])
    elif choice == "2":
        print("Available cities:")
        subprocess.run([sys.executable, "enhanced_data_collection.py", "--list-cities"])
        city = input("Enter city ID: ").strip()
        subprocess.run([sys.executable, "enhanced_data_collection.py", "--city", city])
    elif choice == "3":
        subprocess.run([sys.executable, "enhanced_visualization_app.py"])
    elif choice == "4":
        subprocess.run([sys.executable, "enhanced_data_collection.py", "--list-cities"])
    elif choice == "5":
        print("City configuration feature coming soon!")
        print("For now, edit city_config.py to add new cities")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_start.py", 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("quick_start.py", 0o755)
    
    print("✅ Created quick_start.py script")
    return True

def main():
    """Main installation function"""
    print("🍗 Enhanced BizWizV2 Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Create .env template
    create_env_template()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Validate installation
    if not validate_installation():
        print("❌ Installation validation failed")
        sys.exit(1)
    
    # Create quick start script
    create_quick_start_script()
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Run: python quick_start.py")
    print("3. Or run individual scripts:")
    print("   - python enhanced_data_collection.py --list-cities")
    print("   - python enhanced_data_collection.py --city grand_forks_nd")
    print("   - python enhanced_visualization_app.py")
    
    # Check if API keys are configured
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
        if "YOURAPIHERE" in content:
            print("\n⚠️  Don't forget to configure your API keys in .env file!")

if __name__ == "__main__":
    main()
