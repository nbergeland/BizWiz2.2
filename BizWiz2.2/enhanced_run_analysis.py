# === ENHANCED RUN_ANALYSIS.PY ===
# Save this as: enhanced_run_analysis.py

"""
Enhanced Quick Start Script for BizWizV2.2
This replaces the original run_analysis.py with multi-city support
"""
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required files exist"""
    required_files = [
        "city_config.py",
        "enhanced_data_collection.py", 
        "enhanced_visualization_app.py",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check if .env is configured
    with open(".env", 'r') as f:
        env_content = f.read()
    
    if "YOURAPIHERE" in env_content:
        print("⚠️  Please configure your API keys in .env file before running analysis")
        return False
    
    return True

def main():
    print("🍗 Enhanced BizWizV2 - Multi-City Analysis Tool")
    print("=" * 60)
    
    if not check_dependencies():
        print("\n🔧 Run this first: python install.py")
        return
    
    print("What would you like to do?")
    print("\n📊 Data Collection:")
    print("  1. Analyze current city (check city_configs.yaml)")
    print("  2. Analyze specific city")
    print("  3. List available cities")
    
    print("\n🌐 Visualization:")
    print("  4. Start interactive dashboard")
    
    print("\n⚙️  Configuration:")
    print("  5. Show current configuration")
    print("  6. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\n🔄 Starting data collection for current city...")
            os.system(f"{sys.executable} enhanced_data_collection.py")
            break
            
        elif choice == "2":
            print("\n📍 Available cities:")
            os.system(f"{sys.executable} enhanced_data_collection.py --list-cities")
            city_id = input("\nEnter city ID: ").strip()
            if city_id:
                print(f"\n🔄 Starting data collection for {city_id}...")
                os.system(f"{sys.executable} enhanced_data_collection.py --city {city_id}")
            break
            
        elif choice == "3":
            print("\n📍 Available cities:")
            os.system(f"{sys.executable} enhanced_data_collection.py --list-cities")
            
        elif choice == "4":
            print("\n🚀 Starting visualization dashboard...")
            print("🌐 Your browser should open automatically")
            print("📍 If not, visit: http://127.0.0.1:8050")
            os.system(f"{sys.executable} enhanced_visualization_app.py")
            break
            
        elif choice == "5":
            try:
                from city_config import CityConfigManager
                manager = CityConfigManager()
                current = manager.get_current_config()
                if current:
                    print(f"\n📍 Current city: {current.display_name}")
                    print(f"🆔 City ID: {current.city_id}")
                    print(f"📐 Grid points: {len(current.bounds.get_grid_points())}")
                else:
                    print("\n❌ No current city configuration found")
            except ImportError:
                print("\n❌ Cannot load city configuration")
                
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == '__main__':
    main()
