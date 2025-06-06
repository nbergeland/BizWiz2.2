# 🍗 BizWiz2.2: AI-Powered Commercial Location Intelligence System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**BizWiz** is an advanced commercial location analysis system designed to identify optimal restaurant locations using machine learning, demographic analysis, and competitive intelligence. Originally developed for Raising Cane's expansion analysis, it supports **300+ cities across the continental United States**.


## 🎯 **Key Features**

### 🧠 **AI-Powered Location Scoring**
- **Machine Learning Models** for revenue prediction
- **Multi-factor Analysis** including demographics, traffic, competition
- **Feature Engineering** with 25+ location attributes
- **Cross-validation** for model reliability

### 🗺️ **Comprehensive Geographic Coverage**
- **300+ US Cities** with population > 50,000
- **All 48 Continental States** + Washington DC
- **Smart City Boundaries** based on population density
- **Grid-based Analysis** for systematic coverage

### 📊 **Interactive Visualization Dashboard**
- **Real-time Mapping** with Plotly and Dash
- **Multi-tab Interface** (Map, Analytics, Top Locations, Model Performance)
- **Advanced Filtering** by revenue, competition, demographics
- **Responsive Design** with Bootstrap components

### 🏢 **Competitive Intelligence**
- **Automated Competitor Mapping** (Chick-fil-A, McDonald's, etc.)
- **Market Saturation Analysis** by city size
- **Distance-based Competition Scoring**
- **Fast-casual Market Preference** modeling

### 📈 **Data Integration**
- **Census Demographics** (income, age, population)
- **Commercial Real Estate** data integration
- **Traffic Pattern Analysis** 
- **Zoning Compliance** verification
- **University & Employer** proximity scoring

## 🚀 **Quick Start**

### **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### **Installation**
```bash
# Clone the repository
git clone https://github.com/nbergeland/bizwiz2.2.git
cd bizwiz

# Install dependencies
pip install -r requirements.txt

# Generate city configurations (300+ US cities)
python generate_usa_cities.py

# Run data collection for your first city
python enhanced_data_collection.py --city grand_forks_nd

# Launch the visualization dashboard
python enhanced_visualization_app.py
```

### **Access the Dashboard**
Open your browser to: `http://127.0.0.1:8050`

## 📂 **Project Structure**

```
bizwiz2.2/
├── 📋 Core System
│   ├── city_config.py              # City configuration management
│   ├── enhanced_data_collection.py # Data collection & ML pipeline
│   └── enhanced_visualization_app.py # Interactive dashboard
├── 🗂️ Data & Configuration
│   ├── usa_city_configs.yaml       # 300+ city configurations
│   ├── cache_*/                    # City-specific data cache
│   └── requirements.txt            # Python dependencies
├── 🛠️ Utilities
│   ├── generate_usa_cities.py      # City database generator
│   └── debug_startup.py            # Troubleshooting tools
└── 📚 Documentation
    ├── README.md                   # This file
    ├── docs/                       # Additional documentation
    └── examples/                   # Usage examples
```

## 🔧 **Usage Examples**

### **Analyze a Specific City**
```bash
# Collect data and train model for Austin, TX
python enhanced_data_collection.py --city austin_tx

# List all available cities
python enhanced_data_collection.py --list-cities

# Search for cities
python -c "
from city_config import CityConfigManager
manager = CityConfigManager()
print([c.display_name for c in manager.search_cities('texas')])
"
```

### **Advanced Configuration**
```python
from city_config import CityConfigManager

# Load city manager
manager = CityConfigManager()

# Get cities by state
texas_cities = manager.get_cities_by_state('TX')
print(f"Texas has {len(texas_cities)} cities configured")

# Filter by population
large_cities = manager.get_cities_by_population_range(500000, 10000000)
print(f"Found {len(large_cities)} large cities")

# Set current working city
manager.set_current_city('chicago_il')
config = manager.get_current_config()
print(f"Current city: {config.display_name}")
```

## 🏗️ **Architecture Overview**

### **Data Collection Pipeline**
1. **Geographic Bounds** → Define city analysis area
2. **Grid Generation** → Create systematic location points
3. **Data Enrichment** → Gather demographics, competition, traffic
4. **Feature Engineering** → Create ML-ready features
5. **Model Training** → Random Forest regression
6. **Location Scoring** → Predict revenue potential

### **Machine Learning Features**
- **Demographic Features**: Population density, median income, age distribution
- **Competition Features**: Distance to competitors, market saturation
- **Accessibility Features**: Road access scores, public transit proximity
- **Commercial Features**: Traffic patterns, business density
- **Geographic Features**: Distance to city center, major highways

### **Model Performance**
- **Algorithm**: Random Forest Regression
- **Cross-Validation**: 5-fold CV with temporal splits
- **Metrics**: R², MAE, RMSE
- **Feature Importance**: Automated ranking and selection

## 📊 **Dashboard Features**

### **🗺️ Location Map Tab**
- Interactive map with revenue-scored locations
- Competitor positioning overlay
- Existing store locations
- Zoomable with hover details

### **📈 Analytics Tab**
- Revenue distribution analysis
- Competition vs. performance correlation
- Demographics impact visualization
- Market opportunity identification

### **🏆 Top Locations Tab**
- Ranked list of best opportunities
- Detailed location metrics
- Exportable results table
- Geographic coordinates

### **🔬 Model Performance Tab**
- Model accuracy metrics
- Feature importance rankings
- Cross-validation results
- Parameter optimization details

## 🌍 **Supported Cities**

### **Major Metropolitan Areas**
- New York, Los Angeles, Chicago, Houston, Phoenix
- Philadelphia, San Antonio, San Diego, Dallas, San Jose
- Austin, Jacksonville, Fort Worth, Columbus, Charlotte

### **Regional Centers**
- Seattle, Denver, Washington DC, Boston, Nashville
- Oklahoma City, Portland, Las Vegas, Memphis, Louisville
- Milwaukee, Albuquerque, Tucson, Sacramento, Kansas City

### **Mid-Size Markets**
- Colorado Springs, Omaha, Raleigh, Miami, Virginia Beach
- Minneapolis, Tulsa, Arlington, Tampa, New Orleans
- Cleveland, Bakersfield, Aurora, Anaheim, Riverside

### **Emerging Markets**
- 200+ additional cities with populations 50,000-500,000
- College towns and university markets
- State capitals and regional centers
- Growing suburban markets

*Total: **300+ cities** across the continental United States*

## ⚙️ **Configuration**

### **City-Specific Settings**
Each city includes:
- **Geographic Bounds**: Analysis area definition
- **Demographics**: Expected population/income ranges
- **Market Data**: Universities, major employers, state info
- **Competition**: Primary competitors and saturation factors

### **Model Parameters**
```python
# Example city configuration
{
    "city_id": "austin_tx",
    "display_name": "Austin, TX",
    "bounds": {
        "center_lat": 30.2672,
        "center_lon": -97.7431,
        "grid_spacing": 0.005
    },
    "demographics": {
        "typical_income_range": [45000, 100000],
        "market_saturation_factor": 0.85
    }
}
```

## 🔍 **Advanced Features**

### **Multi-City Analysis**
- Compare opportunities across multiple markets
- Identify expansion patterns and preferences
- Portfolio optimization recommendations
- Risk assessment by market diversity

### **Competitive Intelligence**
- Real-time competitor tracking
- Market entry timing analysis
- Cannibalization risk assessment
- White space opportunity identification

### **Custom Scoring Models**
- Industry-specific parameter tuning
- Regional market adjustments
- Brand-specific preference modeling
- ROI optimization algorithms

## 📋 **Requirements**

### **Python Dependencies**
```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
pyyaml>=6.0
pickle-mixin>=1.0.2
```

### **System Requirements**
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for full city database
- **Network**: Internet connection for data collection

## 🚦 **Getting Started Guide**

### **Step 1: Environment Setup**
```bash
# Create virtual environment
python -m venv bizwiz-env
source bizwiz-env/bin/activate  # Linux/Mac
# or
bizwiz-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Initialize City Database**
```bash
# Generate configurations for all 300+ US cities
python generate_usa_cities.py

# Verify installation
python -c "from city_config import CityConfigManager; print('✅ Setup complete!')"
```

### **Step 3: Collect Your First Dataset**
```bash
# Start with a mid-size city for faster processing
python enhanced_data_collection.py --city grand_forks_nd

# Check data collection progress
ls cache_grand_forks_nd/
```

### **Step 4: Launch Dashboard**
```bash
# Start the interactive dashboard
python enhanced_visualization_app.py

# Open browser to http://127.0.0.1:8050
```

### **Step 5: Explore and Analyze**
1. **Select your city** from the dropdown
2. **Adjust filters** to refine location criteria
3. **Explore the map** to see optimal locations
4. **Review analytics** for market insights
5. **Export top locations** for further analysis

## 🛠️ **Troubleshooting**

### **Common Issues**

**Dashboard Won't Start**
```bash
# Run diagnostics
python debug_startup.py

# Check port availability
lsof -i :8050  # Mac/Linux
netstat -an | find "8050"  # Windows
```

**Data Collection Errors**
```bash
# Verify city configuration
python enhanced_data_collection.py --list-cities

# Test with smaller dataset
python enhanced_data_collection.py --city grand_forks_nd --test-mode
```

**Memory Issues**
- Reduce grid spacing in city config
- Process smaller cities first
- Increase system swap space
- Use batch processing mode

### **Debug Mode**
```bash
# Enable verbose logging
export BIZWIZ_DEBUG=1  # Linux/Mac
set BIZWIZ_DEBUG=1     # Windows

# Run with debug output
python enhanced_data_collection.py --city austin_tx --debug
```

## 🤝 **Contributing**

We welcome contributions!

### **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/nbergeland/BizWiz2.2.git
cd bizwiz2.2

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

### **Areas for Contribution**
- 🌍 **Additional Cities**: International market support
- 🤖 **ML Models**: Advanced algorithms and features
- 📊 **Visualizations**: New chart types and interactions
- 🔌 **Integrations**: Additional data sources and APIs
- 📱 **Mobile Support**: Responsive design improvements
- 🧪 **Testing**: Unit tests and integration tests

## 🙏 **Acknowledgments**

- **OpenStreetMap** for geographic data
- **US Census Bureau** for demographic data
- **Plotly/Dash** for visualization framework
- **Scikit-learn** for machine learning capabilities
- **Restaurant industry partners** for validation and feedback

## 📞 **Support & Contact**

- **Email**: contact@bbsllc.ai 

## 🔮 **Roadmap**

### **Version 2.0 (Q2 2025)**
- 🌍 **International Markets**: Canada, UK, Australia
- 📱 **Mobile App**: Native iOS/Android apps
- 🤖 **Advanced AI**: Deep learning models
- ☁️ **Cloud Deployment**: AWS/Azure hosting options

### **Version 2.5 (Q4 2025)**
- 🏢 **Multi-Brand Support**: Configurable for any restaurant chain
- 📊 **Advanced Analytics**: Predictive market modeling
- 🔗 **API Platform**: Third-party integrations
- 🎯 **Real Estate Integration**: MLS and commercial listings

### **Version 3.0 (2026)**
- 🧠 **AI Recommendations**: Automated expansion planning
- 📈 **Portfolio Optimization**: Multi-location strategy
- 🌐 **Global Markets**: Worldwide city coverage
- 🤝 **Enterprise Features**: Team collaboration tools

---

**Built with ❤️ for data driven decision making**

*Transform your expansion strategy with data-driven location intelligence.*
