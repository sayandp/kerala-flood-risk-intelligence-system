"""
============================================================
src/geocoder.py
Address → Latitude/Longitude conversion for Kerala properties
============================================================

WHY THIS EXISTS:
Banks and home buyers don't know GPS coordinates of a property.
They know addresses like "MG Road, Kochi" or "Survey 234/3, Chalakudy".
This module converts those into precise lat/lon for the ML model.

USES:
- Nominatim (OpenStreetMap) — completely free, no API key needed
- Falls back gracefully if address is ambiguous or not found
- Validates that the result is actually inside Kerala
============================================================
"""

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time


# ─── Kerala Bounding Box ──────────────────────────────────
# These are Kerala's geographic boundaries.
# We use this to validate that geocoded results are actually in Kerala.
KERALA_BOUNDS = {
    'min_lat': 8.0,    # Southernmost point (near Parassala)
    'max_lat': 12.8,   # Northernmost point (near Manjeshwar)
    'min_lon': 74.5,   # Westernmost point (Lakshadweep coast)
    'max_lon': 77.5    # Easternmost point (Western Ghats border)
}


class KeralaGeocoder:
    """
    Converts Kerala property addresses into GPS coordinates.
    
    Example usage:
        geocoder = KeralaGeocoder()
        result = geocoder.geocode("Marine Drive, Kochi")
        print(result['latitude'], result['longitude'])
    """
    
    def __init__(self, user_agent="kerala_flood_risk_app"):
        """
        Initialize the geocoder.
        
        user_agent: Required by Nominatim's terms of service.
                    Just identifies your app — any string works.
        """
        self.geolocator = Nominatim(user_agent=user_agent, timeout=10)
    
    
    def geocode(self, address: str, retries: int = 3) -> dict:
        """
        Convert a Kerala address into latitude/longitude.
        
        Args:
            address: Property address (e.g. "MG Road, Kochi")
            retries: How many times to retry if Nominatim times out
        
        Returns:
            Dictionary with these keys:
                - success: True/False
                - latitude: float (or None if failed)
                - longitude: float (or None if failed)
                - full_address: complete formatted address found
                - error: error message if failed
                - is_in_kerala: True if result is within Kerala bounds
        """
        # Always append "Kerala, India" to make sure we get Kerala results
        # Otherwise "Kochi" might match Kochi in Japan or somewhere else
        search_query = f"{address}, Kerala, India"
        
        for attempt in range(retries):
            try:
                location = self.geolocator.geocode(search_query, exactly_one=True)
                
                # Check if Nominatim found anything
                if location is None:
                    return {
                        'success': False,
                        'latitude': None,
                        'longitude': None,
                        'full_address': None,
                        'error': f"Address not found: {address}",
                        'is_in_kerala': False
                    }
                
                lat = location.latitude
                lon = location.longitude
                
                # Validate the coordinates are actually in Kerala
                in_kerala = self._is_in_kerala(lat, lon)
                
                return {
                    'success': True,
                    'latitude': lat,
                    'longitude': lon,
                    'full_address': location.address,
                    'error': None,
                    'is_in_kerala': in_kerala
                }
            
            except GeocoderTimedOut:
                # Nominatim is slow sometimes — wait and retry
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return {
                    'success': False,
                    'latitude': None,
                    'longitude': None,
                    'full_address': None,
                    'error': "Geocoding service timed out. Try again.",
                    'is_in_kerala': False
                }
            
            except GeocoderServiceError as e:
                return {
                    'success': False,
                    'latitude': None,
                    'longitude': None,
                    'full_address': None,
                    'error': f"Geocoding service error: {str(e)}",
                    'is_in_kerala': False
                }
            
            except Exception as e:
                return {
                    'success': False,
                    'latitude': None,
                    'longitude': None,
                    'full_address': None,
                    'error': f"Unexpected error: {str(e)}",
                    'is_in_kerala': False
                }
    
    
    def reverse_geocode(self, latitude: float, longitude: float) -> dict:
        """
        Convert lat/lon back to a human-readable address.
        Useful for displaying "you searched: [actual location name]"
        
        Args:
            latitude: float between -90 and 90
            longitude: float between -180 and 180
        
        Returns:
            Dictionary with the address details
        """
        try:
            location = self.geolocator.reverse(
                f"{latitude}, {longitude}",
                exactly_one=True,
                language='en'
            )
            
            if location is None:
                return {
                    'success': False,
                    'address': None,
                    'error': "No address found for these coordinates"
                }
            
            return {
                'success': True,
                'address': location.address,
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'address': None,
                'error': str(e)
            }
    
    
    def _is_in_kerala(self, latitude: float, longitude: float) -> bool:
        """
        Check if given coordinates fall inside Kerala state boundaries.
        Internal helper method (underscore prefix = private).
        """
        return (
            KERALA_BOUNDS['min_lat'] <= latitude <= KERALA_BOUNDS['max_lat']
            and
            KERALA_BOUNDS['min_lon'] <= longitude <= KERALA_BOUNDS['max_lon']
        )


# ─── Quick Test (only runs if you execute this file directly) ─────
if __name__ == "__main__":
    """
    To test this file, run from project root:
        python src/geocoder.py
    """
    print("Testing Kerala Geocoder...\n")
    
    geocoder = KeralaGeocoder()
    
    test_addresses = [
        "Marine Drive, Kochi",
        "Trivandrum Central Railway Station",
        "Chalakudy, Thrissur",
        "Kuttanad, Alappuzha",
        "Random nonsense address xyz123"
    ]
    
    for address in test_addresses:
        print(f"Searching: {address}")
        result = geocoder.geocode(address)
        
        if result['success']:
            print(f"  ✓ Found: {result['latitude']:.4f}, {result['longitude']:.4f}")
            print(f"    Address: {result['full_address'][:80]}...")
            print(f"    In Kerala: {result['is_in_kerala']}")
        else:
            print(f"  ✗ Failed: {result['error']}")
        
        print()
        time.sleep(1)  # Be polite to free Nominatim service