# === ENHANCED MULTI-CITY DATA COLLECTION SCRIPT ===
# Save this as: enhanced_data_collection.py

import os
import numpy as np
import pandas as pd
import googlemaps
import requests
import time
import json
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import radians, cos, sin, asin, sqrt
import pickle
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our new city configuration system
from city_config import CityConfigManager, CityConfiguration



# Debug before loading
print(f"Current working directory: {os.getcwd()}")
print(f".env file exists: {os.path.exists('.env')}")



# Check what's in the environment
print(f"All env vars with GOOGLE: {[k for k in os.environ.keys() if 'GOOGLE' in k.upper()]}")

CENSUS_API_KEY='YOURAPIHERE'
GOOGLE_API_KEY='YOURAPIHERE'
RENTCAST_API_KEY='YOURAPIHERE' 

print(f"GOOGLE_API_KEY value: '{GOOGLE_API_KEY}'")
print(f"CENSUS_API_KEY value: '{CENSUS_API_KEY}'")
# === GOOGLE MAPS CLIENT ===
gmaps = googlemaps.Client(key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

# === ENHANCED CACHING SYSTEM ===
class EnhancedCacheManager:
    """Enhanced caching system with city-specific caches"""
    
    def __init__(self, city_id: str):
        self.city_id = city_id
        self.cache_dir = f"cache_{city_id}"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(self.cache_dir, 'location_data_cache.pkl')
        self.usage_file = os.path.join(self.cache_dir, 'api_usage.json')
        self.processed_data_file = os.path.join(self.cache_dir, 'processed_location_data.pkl')
        self.model_metrics_file = os.path.join(self.cache_dir, 'model_metrics.json')
        
    def load_cache(self):
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache, f)

    def load_api_usage(self):
        try:
            with open(self.usage_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'daily_calls': 0, 'date': str(datetime.date.today())}

    def save_api_usage(self, usage):
        with open(self.usage_file, 'w') as f:
            json.dump(usage, f)
            
    def save_model_metrics(self, metrics):
        with open(self.model_metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def load_model_metrics(self):
        try:
            with open(self.model_metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def track_api_call(self):
        """Track API calls to monitor usage"""
        usage = self.load_api_usage()
        today = str(datetime.date.today())
        
        if usage['date'] != today:
            usage = {'daily_calls': 0, 'date': today}
        
        usage['daily_calls'] += 1
        self.save_api_usage(usage)
        
        print(f"API calls today for {self.city_id}: {usage['daily_calls']}")
        return usage['daily_calls']

# === REAL ZONING DATA INTEGRATION ===
class ZoningDataFetcher:
    """Fetches real zoning data from various sources"""
    
    def __init__(self, city_config: CityConfiguration, cache_manager: EnhancedCacheManager):
        self.city_config = city_config
        self.cache_manager = cache_manager
        self.cache = cache_manager.load_cache()
    
    def get_zoning_compliance(self, lat: float, lon: float) -> bool:
        """Get real zoning compliance for a location"""
        cache_key = f'zoning_{lat:.4f}_{lon:.4f}'
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try multiple zoning data sources
        compliance = self._check_overpass_zoning(lat, lon)
        if compliance is None:
            compliance = self._check_government_apis(lat, lon)
        if compliance is None:
            compliance = self._estimate_zoning_from_nearby_pois(lat, lon)
        
        # Default to True if we can't determine (conservative approach)
        if compliance is None:
            compliance = True
            
        self.cache[cache_key] = compliance
        self.cache_manager.save_cache(self.cache)
        return compliance
    
    def _check_overpass_zoning(self, lat: float, lon: float) -> Optional[bool]:
        """Check zoning using OpenStreetMap data"""
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:json][timeout:10];
            (
              way["landuse"~"commercial|retail|mixed"]({lat-0.001},{lon-0.001},{lat+0.001},{lon+0.001});
              relation["landuse"~"commercial|retail|mixed"]({lat-0.001},{lon-0.001},{lat+0.001},{lon+0.001});
            );
            out geom;
            """
            
            response = requests.get(overpass_url, params={'data': overpass_query}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return len(data.get('elements', [])) > 0
        except:
            pass
        return None
    
    def _check_government_apis(self, lat: float, lon: float) -> Optional[bool]:
        """Check government zoning APIs (placeholder for real implementations)"""
        # This would integrate with specific city/county zoning APIs
        # For now, we'll implement a realistic heuristic
        
        # Check if near known commercial areas using Google Places
        if gmaps:
            try:
                result = gmaps.places_nearby(
                    location=(lat, lon),
                    radius=200,  # 200 meter radius
                    type='establishment'
                )
                
                commercial_types = ['store', 'restaurant', 'gas_station', 'bank', 'shopping_mall']
                commercial_places = [
                    place for place in result.get('results', [])
                    if any(place_type in place.get('types', []) for place_type in commercial_types)
                ]
                
                # If there are 2+ commercial establishments nearby, likely zoned commercial
                return len(commercial_places) >= 2
                
            except Exception as e:
                print(f"Error checking government APIs: {e}")
        
        return None
    
    def _estimate_zoning_from_nearby_pois(self, lat: float, lon: float) -> Optional[bool]:
        """Estimate zoning based on nearby POIs"""
        # This is a fallback method using the POI data we already collect
        # We'll implement this in the main data fetcher
        return None

# === DISTANCE FUNCTION IN MILES ===
def calculate_distance_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    c = 2 * asin(sqrt(a))
    return c * 3956

# === ENHANCED ML MODEL PIPELINE ===
class EnhancedMLPipeline:
    """Enhanced machine learning pipeline with validation and metrics"""
    
    def __init__(self, city_config: CityConfiguration, cache_manager: EnhancedCacheManager):
        self.city_config = city_config
        self.cache_manager = cache_manager
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        
    def train_and_validate_model(self, df: pd.DataFrame) -> Dict:
        """Train model with proper validation and return metrics"""
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in [
            'latitude', 'longitude', 'estimated_revenue', 'keep_location'
        ]]
        
        X = df[feature_columns].select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        y = df['estimated_revenue']
        
        print(f"Training model with {len(X)} samples and {len(X.columns)} features")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, 
            scoring='neg_mean_absolute_error'
        )
        
        # Final predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        self.metrics = {
            'best_parameters': grid_search.best_params_,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'train_mae': mean_absolute_error(y, y_pred),
            'train_mse': mean_squared_error(y, y_pred),
            'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'train_r2': r2_score(y, y_pred),
            'feature_count': len(X.columns),
            'sample_count': len(X),
            'city_id': self.city_config.city_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save metrics
        self.cache_manager.save_model_metrics(self.metrics)
        
        print(f"\nModel Performance for {self.city_config.display_name}:")
        print(f"Cross-validation MAE: ${self.metrics['cv_mae_mean']:,.0f} ± ${self.metrics['cv_mae_std']:,.0f}")
        print(f"Training R²: {self.metrics['train_r2']:.3f}")
        print(f"Training RMSE: ${self.metrics['train_rmse']:,.0f}")
        
        return self.metrics

# === ENHANCED DATA FETCHER ===
class EnhancedCommercialLocationDataFetcher:
    def __init__(self, city_config: CityConfiguration):
        self.city_config = city_config
        self.cache_manager = EnhancedCacheManager(city_config.city_id)
        self.cache = self.cache_manager.load_cache()
        self.zoning_fetcher = ZoningDataFetcher(city_config, self.cache_manager)
        
        # Initialize data containers
        self.chickfila_locations = None
        self.raising_canes_locations = None
        self.competitor_locations = {}
        self.poi_locations = {}
        self.active_listings = []
        self.road_points = []
        
    def fetch_all_chickfila_locations(self):
        """Fetch all Chick-fil-A locations in the broader area once"""
        if self.chickfila_locations is not None:
            return
            
        cache_key = f'chickfila_all_{self.city_config.city_id}'
        if cache_key in self.cache:
            self.chickfila_locations = self.cache[cache_key]
            return
            
        bounds = self.city_config.bounds
        
        try:
            self.cache_manager.track_api_call()
            result = gmaps.places_nearby(
                location=(bounds.center_lat, bounds.center_lon), 
                radius=50000,  # 50km radius
                keyword=self.city_config.competitor_data.primary_competitor
            )
            locations = result['results']
            
            # Handle pagination if needed
            while 'next_page_token' in result:
                time.sleep(2)
                self.cache_manager.track_api_call()
                result = gmaps.places_nearby(
                    location=(bounds.center_lat, bounds.center_lon),
                    radius=50000,
                    keyword=self.city_config.competitor_data.primary_competitor,
                    page_token=result['next_page_token']
                )
                locations.extend(result['results'])
                
            self.chickfila_locations = [(
                loc['geometry']['location']['lat'],
                loc['geometry']['location']['lng']
            ) for loc in locations]
            
            self.cache[cache_key] = self.chickfila_locations
            self.cache_manager.save_cache(self.cache)
            
        except Exception as e:
            print(f"Error fetching {self.city_config.competitor_data.primary_competitor} locations: {e}")
            self.chickfila_locations = []

    def fetch_all_raising_canes_locations(self):
        """Fetch all Raising Cane's locations in the broader area once"""
        if self.raising_canes_locations is not None:
            return
            
        cache_key = f'raising_canes_all_{self.city_config.city_id}'
        if cache_key in self.cache:
            self.raising_canes_locations = self.cache[cache_key]
            return
            
        bounds = self.city_config.bounds
        
        try:
            self.cache_manager.track_api_call()
            result = gmaps.places_nearby(
                location=(bounds.center_lat, bounds.center_lon), 
                radius=50000,  # 50km radius
                keyword="raising cane's"
            )
            locations = result['results']
            
            # Handle pagination if needed
            while 'next_page_token' in result:
                time.sleep(2)
                self.cache_manager.track_api_call()
                result = gmaps.places_nearby(
                    location=(bounds.center_lat, bounds.center_lon),
                    radius=50000,
                    keyword="raising cane's",
                    page_token=result['next_page_token']
                )
                locations.extend(result['results'])
                
            self.raising_canes_locations = [(
                loc['geometry']['location']['lat'],
                loc['geometry']['location']['lng'],
                loc.get('name', "Raising Cane's")
            ) for loc in locations]
            
            self.cache[cache_key] = self.raising_canes_locations
            self.cache_manager.save_cache(self.cache)
            
        except Exception as e:
            print(f"Error fetching Raising Cane's locations: {e}")
            self.raising_canes_locations = []
    
    def fetch_competitor_locations(self):
        """Fetch all competitor locations once"""
        if self.competitor_locations:
            return
            
        bounds = self.city_config.bounds
        competitors = self.city_config.competitor_data.competitor_search_terms
        
        for competitor in competitors:
            cache_key = f'competitor_{competitor}_{self.city_config.city_id}'
            if cache_key in self.cache:
                self.competitor_locations[competitor] = self.cache[cache_key]
                continue
                
            try:
                self.cache_manager.track_api_call()
                result = gmaps.places_nearby(
                    location=(bounds.center_lat, bounds.center_lon),
                    radius=20000,  # 20km radius
                    keyword=competitor
                )
                locations = [(
                    loc['geometry']['location']['lat'],
                    loc['geometry']['location']['lng']
                ) for loc in result['results']]
                
                self.competitor_locations[competitor] = locations
                self.cache[cache_key] = locations
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching {competitor} locations: {e}")
                self.competitor_locations[competitor] = []
        
        self.cache_manager.save_cache(self.cache)
    
    def fetch_commercial_poi_locations(self):
        """Fetch commercial-focused points of interest"""
        if self.poi_locations:
            return
            
        bounds = self.city_config.bounds
        
        # Commercial-focused POI with higher weights for business viability
        poi_types = [
            ('shopping_mall', 'shopping_mall', 50),      # High traffic generators
            ('gas_station', 'gas_station', 30),         # High visibility locations
            ('bank', 'bank', 25),                       # Commercial corridors
            ('pharmacy', 'pharmacy', 20),               # Strip mall locations
            ('supermarket', 'supermarket', 40),         # Anchor stores
            ('hospital', 'hospital', 30),               # High traffic
            ('university', 'university', 35),           # Student traffic
            ('gym', 'gym', 15),                         # Commercial areas
            ('car_dealer', 'car_dealer', 20),           # Commercial strips
            ('lodging', 'lodging', 25),                 # Commercial zones
            ('store', 'store', 10),                     # General retail
            ('restaurant', 'restaurant', 5)             # Food service areas
        ]
        
        for poi_name, poi_type, weight in poi_types:
            cache_key = f'poi_{poi_name}_{self.city_config.city_id}'
            if cache_key in self.cache:
                self.poi_locations[poi_name] = self.cache[cache_key]
                continue
                
            try:
                self.cache_manager.track_api_call()
                if poi_name == 'university':
                    # Use specific university names for this city
                    all_locations = []
                    for university in self.city_config.market_data.major_universities:
                        result = gmaps.places_nearby(
                            location=(bounds.center_lat, bounds.center_lon),
                            radius=25000,
                            keyword=university
                        )
                        all_locations.extend(result['results'])
                    locations = [(
                        loc['geometry']['location']['lat'],
                        loc['geometry']['location']['lng'],
                        weight
                    ) for loc in all_locations]
                else:
                    result = gmaps.places_nearby(
                        location=(bounds.center_lat, bounds.center_lon),
                        radius=25000,
                        type=poi_type
                    )
                    locations = [(
                        loc['geometry']['location']['lat'],
                        loc['geometry']['location']['lng'],
                        weight
                    ) for loc in result['results']]
                
                self.poi_locations[poi_name] = locations
                self.cache[cache_key] = locations
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error fetching {poi_name} locations: {e}")
                self.poi_locations[poi_name] = []
        
        self.cache_manager.save_cache(self.cache)

    def fetch_rental_listings(self):
        """Fetch rental listings once"""
        if self.active_listings:
            return
            
        cache_key = f'rental_listings_{self.city_config.city_id}'
        if cache_key in self.cache:
            # Check if cache is recent (less than 24 hours old)
            cache_time = self.cache.get(f'{cache_key}_timestamp', 0)
            if time.time() - cache_time < 86400:  # 24 hours
                self.active_listings = self.cache[cache_key]
                return
        
        try:
            url = "https://api.rentcast.io/v1/listings/rental/long-term"
            headers = {"X-Api-Key": RENTCAST_API_KEY}
            
            # Use city-specific rental API name
            params = {
                "city": self.city_config.market_data.rental_api_city_name, 
                "state": self.city_config.market_data.state_code, 
                "status": "active", 
                "limit": 500
            }
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                self.active_listings = response.json().get('listings', [])
                self.cache[cache_key] = self.active_listings
                self.cache[f'{cache_key}_timestamp'] = time.time()
                self.cache_manager.save_cache(self.cache)
            else:
                print(f"Rental API returned status {response.status_code}")
                self.active_listings = []
                
        except Exception as e:
            print(f"Error fetching rental listings: {e}")
            self.active_listings = []

    def fetch_road_data(self):
        """Load road data from OpenStreetMap"""
        cache_key = f'osm_roads_{self.city_config.city_id}'
        
        if cache_key in self.cache:
            self.road_points = self.cache[cache_key]
            return
        
        bounds = self.city_config.bounds
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:25];
        (
          way["highway"~"^(trunk|primary|secondary|trunk_link|primary_link)$"]({bounds.min_lat},{bounds.min_lon},{bounds.max_lat},{bounds.max_lon});
        );
        out geom;
        """
        
        try:
            print(f"Fetching road data for {self.city_config.display_name}...")
            response = requests.get(overpass_url, params={'data': overpass_query})
            roads_data = response.json()
            
            # Extract road coordinates
            road_points = []
            for way in roads_data.get('elements', []):
                if 'geometry' in way:
                    for point in way['geometry']:
                        road_points.append((point['lat'], point['lon']))
            
            self.road_points = road_points
            self.cache[cache_key] = road_points
            self.cache_manager.save_cache(self.cache)
            print(f"Found {len(road_points)} road points")
            
        except Exception as e:
            print(f"Error fetching road data: {e}")
            self.road_points = []

    @lru_cache(maxsize=1000)
    def get_demographics_cached(self, lat_rounded, lon_rounded):
        """Cache demographics by rounded coordinates to avoid duplicate census calls"""
        try:
            fcc_url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat_rounded}&longitude={lon_rounded}&format=json"
            fcc_response = requests.get(fcc_url)
            fips = fcc_response.json()['Block']['FIPS'][:11]
            
            url = f"https://api.census.gov/data/2020/acs/acs5?get=B01003_001E,B19013_001E,B01002_001E&for=tract:{fips[5:11]}&in=state:{fips[:2]}+county:{fips[2:5]}&key={CENSUS_API_KEY}"
            data = requests.get(url).json()[1]
            
            return {
                'population': int(data[0]), 
                'median_income': int(data[1]), 
                'median_age': float(data[2])
            }
        except:
            # Use city-specific defaults
            demo = self.city_config.demographics
            return {
                'population': int(np.mean(demo.typical_population_range)), 
                'median_income': int(np.mean(demo.typical_income_range)), 
                'median_age': float(np.mean(demo.typical_age_range))
            }

    def calculate_commercial_viability_score(self, lat, lon):
        """Calculate commercial viability without additional API calls"""
        
        # Commercial foot traffic score (much higher weights)
        commercial_traffic = 0
        for poi_type, locations in self.poi_locations.items():
            if poi_type in ['shopping_mall', 'supermarket', 'gas_station', 'bank']:
                for p_lat, p_lon, weight in locations:
                    distance = calculate_distance_miles(lat, lon, p_lat, p_lon)
                    if distance <= 1:  # Within 1 mile
                        commercial_traffic += weight
        
        # Visibility/accessibility score using pre-fetched road data
        road_accessibility = 0
        if self.road_points:
            nearby_roads = [
                1 for r_lat, r_lon in self.road_points
                if calculate_distance_miles(lat, lon, r_lat, r_lon) <= 0.2  # Within 0.2 miles
            ]
            road_accessibility = min(len(nearby_roads) * 5, 50)  # Cap at 50
        
        # Gas station proximity (indicator of major roads and visibility)
        gas_stations = self.poi_locations.get('gas_station', [])
        gas_proximity = 0
        for g_lat, g_lon, _ in gas_stations:
            distance = calculate_distance_miles(lat, lon, g_lat, g_lon)
            if distance <= 0.5:  # Within 0.5 miles
                gas_proximity += 15
        
        return {
            'commercial_traffic_score': commercial_traffic,
            'road_accessibility_score': road_accessibility,
            'gas_station_proximity': gas_proximity
        }

    def detect_residential_bias(self, lat, lon, active_listings_count, population):
        """Detect if location is heavily residential"""
        
        residential_indicators = 0
        
        # Adjust thresholds based on city demographics
        demo = self.city_config.demographics
        high_listing_threshold = 15 * demo.population_density_factor
        high_pop_threshold = demo.typical_population_range[1] * 0.8
        
        # High apartment density
        if active_listings_count > high_listing_threshold:
            residential_indicators += 10
        
        # Very high population density (typical of residential areas)
        if population > high_pop_threshold:
            residential_indicators += 15
        
        # Low commercial activity
        commercial_nearby = 0
        for poi_type in ['gas_station', 'bank', 'shopping_mall']:
            locations = self.poi_locations.get(poi_type, [])
            for p_lat, p_lon, _ in locations:
                if calculate_distance_miles(lat, lon, p_lat, p_lon) <= 0.5:
                    commercial_nearby += 1
        
        if commercial_nearby == 0:
            residential_indicators += 10
        
        return residential_indicators

    def calculate_features_for_point(self, lat, lon):
        """Calculate all features for a single point using cached data"""
        # Round coordinates for demographic caching
        lat_rounded = round(lat, 3)
        lon_rounded = round(lon, 3)
        
        # Primary competitor (Chick-fil-A) proximity
        if self.chickfila_locations:
            distances_to_chickfila = [
                calculate_distance_miles(lat, lon, c_lat, c_lon) 
                for c_lat, c_lon in self.chickfila_locations
            ]
            chick_count = len([d for d in distances_to_chickfila if d <= 5])
            nearest_chickfila = min(distances_to_chickfila) if distances_to_chickfila else 30
        else:
            chick_count, nearest_chickfila = 0, 30
        
        # Fast food competition
        competition_count = 0
        for competitor, locations in self.competitor_locations.items():
            nearby_competitors = [
                1 for c_lat, c_lon in locations 
                if calculate_distance_miles(lat, lon, c_lat, c_lon) <= 2
            ]
            competition_count += len(nearby_competitors)
        
        # Commercial viability scores
        commercial_scores = self.calculate_commercial_viability_score(lat, lon)
        
        # Demographics (cached)
        demographics = self.get_demographics_cached(lat_rounded, lon_rounded)
        
        # Rental data
        nearby_listings = []
        for listing in self.active_listings:
            if listing.get('latitude') and listing.get('longitude'):
                distance = calculate_distance_miles(
                    lat, lon, listing['latitude'], listing['longitude']
                )
                if distance <= 1:
                    nearby_listings.append(listing['price'])
        
        active_listings_count = len(nearby_listings)
        avg_rent = np.mean(nearby_listings) if nearby_listings else 0
        
        # Residential bias detection
        residential_bias = self.detect_residential_bias(
            lat, lon, active_listings_count, demographics['population']
        )
        
        # Real zoning compliance
        zoning = self.zoning_fetcher.get_zoning_compliance(lat, lon)
        
        return {
            'latitude': lat,
            'longitude': lon,
            'chickfila_count_nearby': chick_count,
            'distance_to_chickfila': nearest_chickfila,
            'fast_food_competition': competition_count,
            'commercial_traffic_score': commercial_scores['commercial_traffic_score'],
            'road_accessibility_score': commercial_scores['road_accessibility_score'],
            'gas_station_proximity': commercial_scores['gas_station_proximity'],
            'population': demographics['population'],
            'median_income': demographics['median_income'],
            'median_age': demographics['median_age'],
            'rent_per_sqft': 12.50,
            'zoning_compliant': int(zoning),
            'active_listings_within_1_mile': active_listings_count,
            'average_nearby_rent': avg_rent,
            'residential_bias_score': residential_bias,
            'market_saturation_factor': self.city_config.competitor_data.market_saturation_factor,
            'fast_casual_preference': self.city_config.competitor_data.fast_casual_preference_score
        }

# === MAIN DATA COLLECTION FUNCTION ===
def collect_and_process_all_data(city_id: str = None):
    """Main function to collect all data and save processed results"""
    
    # Initialize city configuration
    city_manager = CityConfigManager()
    
    if city_id:
        if not city_manager.set_current_city(city_id):
            print(f"City ID '{city_id}' not found. Available cities: {city_manager.list_cities()}")
            return None
    
    city_config = city_manager.get_current_config()
    if not city_config:
        print("No city configuration found!")
        return None
    
    print(f"Starting analysis for {city_config.display_name}")
    
    # Initialize cache manager
    cache_manager = EnhancedCacheManager(city_config.city_id)
    
    # Check if processed data already exists
    if os.path.exists(cache_manager.processed_data_file):
        print("Processed data file already exists. Loading existing data...")
        with open(cache_manager.processed_data_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    print("Starting commercial location analysis...")
    fetcher = EnhancedCommercialLocationDataFetcher(city_config)
    
    print("Fetching bulk data...")
    fetcher.fetch_all_chickfila_locations()
    print(f"Found {len(fetcher.chickfila_locations)} {city_config.competitor_data.primary_competitor} locations")
    
    fetcher.fetch_all_raising_canes_locations()
    print(f"Found {len(fetcher.raising_canes_locations)} Raising Cane's locations")
    
    fetcher.fetch_competitor_locations()
    total_competitors = sum(len(locs) for locs in fetcher.competitor_locations.values())
    print(f"Found {total_competitors} competitor locations")
    
    fetcher.fetch_commercial_poi_locations()
    total_pois = sum(len(locs) for locs in fetcher.poi_locations.values())
    print(f"Found {total_pois} points of interest")
    
    fetcher.fetch_rental_listings()
    print(f"Found {len(fetcher.active_listings)} rental listings")
    
    fetcher.fetch_road_data()
    print(f"Found {len(fetcher.road_points)} road points")
    
    # Get grid points for this city
    grid_points = city_config.bounds.get_grid_points()
    print(f"Processing {len(grid_points)} grid points...")
    
    feature_list = []
    
    for idx, (lat, lon) in enumerate(grid_points):
        if idx % 10 == 0:
            print(f"Processing {idx+1}/{len(grid_points)}: {lat:.4f}, {lon:.4f}")
        
        features = fetcher.calculate_features_for_point(lat, lon)
        feature_list.append(features)
        
        # Minimal delay
        if idx % 100 == 0:
            time.sleep(0.1)
    
    df = pd.DataFrame(feature_list)
    
    # Calculate derived commercial features with city-specific adjustments
    df['chick_fil_a_advantage'] = np.where(
        (df['distance_to_chickfila'] > 2) & (df['distance_to_chickfila'] < 8), 
        800 / df['distance_to_chickfila'] * city_config.competitor_data.market_saturation_factor, 
        0
    )

    # Youth factor (important for fast-casual dining)
    youth_threshold = city_config.demographics.typical_age_range[1] * 0.8
    df['youth_factor'] = np.where(df['median_age'] < youth_threshold, 
                                  800 * city_config.competitor_data.fast_casual_preference_score, 0)

    # Competition clustering advantage (some competition is good)
    df['competitive_cluster_bonus'] = np.where(
        (df['fast_food_competition'] >= 2) & (df['fast_food_competition'] <= 6), 
        300 * city_config.competitor_data.market_saturation_factor, 
        np.where(df['fast_food_competition'] > 6, -200, 0)
    )

    # City-specific income adjustment
    income_multiplier = city_config.demographics.typical_income_range[1] / 70000  # Normalize to Grand Forks baseline
    
    # Commercial-focused revenue calculation with city adjustments
    df['estimated_revenue'] = (
        # Commercial factors (high weights)
        df['commercial_traffic_score'] * 150 +
        df['road_accessibility_score'] * 100 +
        df['gas_station_proximity'] * 80 +
        df['competitive_cluster_bonus'] +
        
        # Demographics (moderate weights, focus on income) with city adjustment
        df['median_income'] * 0.002 * income_multiplier +
        df['youth_factor'] +
        
        # Strategic positioning
        df['chick_fil_a_advantage'] * 400 +
        
        # Penalties for residential bias (adjusted for city density)
        df['active_listings_within_1_mile'] * -100 * city_config.demographics.population_density_factor +
        df['residential_bias_score'] * -150 +
        np.where(df['population'] > city_config.demographics.typical_population_range[1] * 0.9, -500, 0) +
        
        # Zoning compliance bonus
        df['zoning_compliant'] * 1200 +
        
        # Market preference adjustment
        df['fast_casual_preference'] * 500 +
        
        # Base commercial viability
        2000
    )

    # Ensure non-negative revenue
    df['estimated_revenue'] = np.maximum(df['estimated_revenue'], 0)

    # Filter out locations with high residential bias unless they have strong commercial indicators
    df['keep_location'] = (
        (df['residential_bias_score'] < 20) |  # Low residential bias, or
        (df['commercial_traffic_score'] > 50)  # Strong commercial indicators
    )

    # Apply filter
    df_filtered = df[df['keep_location']].copy()

    print(f"Filtered out {len(df) - len(df_filtered)} residential-biased locations")
    print(f"Kept {len(df_filtered)} commercially viable locations")

    # Enhanced ML Pipeline
    ml_pipeline = EnhancedMLPipeline(city_config, cache_manager)
    metrics = ml_pipeline.train_and_validate_model(df_filtered)
    
    # Add predictions to dataframe
    feature_columns = [col for col in df_filtered.columns if col not in [
        'latitude', 'longitude', 'estimated_revenue', 'keep_location'
    ]]
    X = df_filtered[feature_columns].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    df_filtered['predicted_revenue'] = ml_pipeline.model.predict(X)

    print(f"\nCommercial location analysis complete for {city_config.display_name}")
    print(f"Processed {len(df_filtered)} locations.")
    print(f"Top predicted revenue: ${df_filtered['predicted_revenue'].max():,.0f}")
    
    # Save all processed data
    processed_data = {
        'df_filtered': df_filtered,
        'model': ml_pipeline.model,
        'feature_importance': ml_pipeline.feature_importance,
        'metrics': metrics,
        'chickfila_locations': fetcher.chickfila_locations,
        'raising_canes_locations': fetcher.raising_canes_locations,
        'city_config': city_config,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(cache_manager.processed_data_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed data saved to {cache_manager.processed_data_file}")
    
    return processed_data

# === CLI INTERFACE ===
def main():
    """Command line interface for the enhanced data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Commercial Location Analysis')
    parser.add_argument('--city', type=str, help='City ID to analyze (e.g., grand_forks_nd)')
    parser.add_argument('--list-cities', action='store_true', help='List available cities')
    
    args = parser.parse_args()
    
    city_manager = CityConfigManager()
    
    if args.list_cities:
        print("Available cities:")
        for city_id in city_manager.list_cities():
            config = city_manager.get_config(city_id)
            current = " (current)" if city_id == city_manager.current_city else ""
            print(f"  {city_id}: {config.display_name}{current}")
        return
    
    if args.city:
        if args.city not in city_manager.list_cities():
            print(f"City '{args.city}' not found. Use --list-cities to see available cities.")
            return
        result = collect_and_process_all_data(args.city)
    else:
        result = collect_and_process_all_data()
    
    if result:
        print("Analysis completed successfully!")
    else:
        print("Analysis failed!")

# Run data collection if this script is executed directly
if __name__ == '__main__':
    main()
