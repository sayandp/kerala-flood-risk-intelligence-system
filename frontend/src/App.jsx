import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, MapPin, Mountain, Droplets, TreePine, TrendingUp, 
  AlertTriangle, Shield, IndianRupee, Sparkles, Loader2,
  ChevronRight, Activity, Cloud, Compass, Crosshair, Map as MapIcon,
  Pin, Globe, Zap, ArrowUpRight, Flame, EyeOff, Eye
} from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ─── Severity Color Map ──────────────────────────────────
const SEVERITY_STYLES = {
  'VERY LOW': { gradient: 'from-emerald-400 to-green-500', text: 'text-emerald-400', glow: 'shadow-emerald-500/40', accent: '#10b981' },
  'LOW':      { gradient: 'from-lime-400 to-emerald-500',  text: 'text-lime-400',    glow: 'shadow-lime-500/40',    accent: '#84cc16' },
  'MODERATE': { gradient: 'from-amber-400 to-orange-500',  text: 'text-amber-400',   glow: 'shadow-amber-500/40',   accent: '#f59e0b' },
  'HIGH':     { gradient: 'from-orange-400 to-red-500',    text: 'text-orange-400',  glow: 'shadow-orange-500/40',  accent: '#f97316' },
  'CRITICAL': { gradient: 'from-red-500 to-pink-600',      text: 'text-red-400',     glow: 'shadow-red-500/50',     accent: '#dc2626' },
};

const EXAMPLE_ADDRESSES = [
  { label: 'Kuttanad', value: 'Kuttanad, Alappuzha', emoji: '🚨' },
  { label: 'Aluva',    value: 'Aluva, Ernakulam',     emoji: '🟠' },
  { label: 'Munnar',   value: 'Munnar, Idukki',       emoji: '✅' },
  { label: 'Wayanad',  value: 'Wayanad, Kerala',      emoji: '✅' },
];

// Custom marker
const createCustomIcon = (color) => L.divIcon({
  className: 'custom-marker',
  html: `<div style="position:relative;width:32px;height:32px;">
    <div style="position:absolute;top:-12px;left:-12px;width:56px;height:56px;background:${color};border-radius:50%;opacity:0.3;animation:pulse-ring 2s ease-out infinite;"></div>
    <div style="position:absolute;top:-6px;left:-6px;width:44px;height:44px;background:${color};border-radius:50%;opacity:0.5;animation:pulse-ring 2s ease-out infinite;animation-delay:0.5s;"></div>
    <div style="position:relative;width:32px;height:32px;background:${color};border:3px solid white;border-radius:50%;box-shadow:0 4px 12px rgba(0,0,0,0.5);"></div>
  </div>`,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
});

const PIN_ICON = L.divIcon({
  className: 'custom-marker',
  html: `<div style="position:relative;width:36px;height:36px;">
    <div style="position:absolute;top:-12px;left:-12px;width:60px;height:60px;background:#a78bfa;border-radius:50%;opacity:0.2;animation:pulse-ring 1.5s ease-out infinite;"></div>
    <div style="position:relative;width:36px;height:36px;background:linear-gradient(135deg, #a78bfa, #ec4899);border:3px solid white;border-radius:50%;box-shadow:0 6px 16px rgba(167,139,250,0.6);display:flex;align-items:center;justify-content:center;color:white;font-size:18px;">📍</div>
  </div>`,
  iconSize: [36, 36],
  iconAnchor: [18, 18],
});

function getTypeIcon(type) {
  const t = (type || '').toLowerCase();
  if (t === 'house' || t === 'residential') return '🏠';
  if (t === 'town' || t === 'city' || t === 'village') return '🏙️';
  if (t === 'stream' || t === 'river' || t === 'water') return '🌊';
  if (t === 'pond' || t === 'lake' || t === 'reservoir') return '💧';
  if (t === 'administrative') return '📍';
  if (t === 'highway' || t === 'road') return '🛣️';
  return '📌';
}


// ─── Main App ────────────────────────────────────────────
export default function App() {
  const [searchMode, setSearchMode] = useState('address');
  const [address, setAddress] = useState('');
  const [coords, setCoords] = useState({ lat: '10.0', lon: '76.5' });
  const [pickedLocation, setPickedLocation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loadingStage, setLoadingStage] = useState('');
  const [riskGrid, setRiskGrid] = useState(null);

  // Load risk grid once on mount (in background)
  useEffect(() => {
    fetch(`${API_URL}/api/risk-grid`)
      .then(res => res.json())
      .then(data => setRiskGrid(data.points || []))
      .catch(err => console.warn('Risk grid not available:', err));
  }, []);

  const handleSearch = async (overrideAddress) => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    const stages = [
      'Locating property...',
      'Extracting satellite features...',
      'Running ML model...',
      'Applying safety rules...',
      'Generating projections...',
    ];
    
    let stageIdx = 0;
    setLoadingStage(stages[0]);
    const stageInterval = setInterval(() => {
      stageIdx = (stageIdx + 1) % stages.length;
      setLoadingStage(stages[stageIdx]);
    }, 800);
    
    try {
      let response;
      const addressToUse = typeof overrideAddress === 'string' ? overrideAddress : address;
      
      if (searchMode === 'address') {
        if (!addressToUse.trim()) throw new Error('Enter an address');
        response = await fetch(`${API_URL}/api/predict-by-address`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ address: addressToUse }),
        });
      } else if (searchMode === 'coords') {
        const lat = parseFloat(coords.lat);
        const lon = parseFloat(coords.lon);
        if (isNaN(lat) || isNaN(lon)) throw new Error('Invalid coordinates');
        if (lat < 8.0 || lat > 12.8 || lon < 74.5 || lon > 77.5) {
          throw new Error('Coordinates outside Kerala (lat: 8-12.8, lon: 74.5-77.5)');
        }
        response = await fetch(`${API_URL}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ latitude: lat, longitude: lon }),
        });
      } else if (searchMode === 'map') {
        if (!pickedLocation) throw new Error('Click on the map to pick a location');
        response = await fetch(`${API_URL}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ latitude: pickedLocation.lat, longitude: pickedLocation.lng }),
        });
      }
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze property');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(stageInterval);
      setLoading(false);
      setLoadingStage('');
    }
  };

  const handleSearchByCoords = async (lat, lon) => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    const stages = [
      'Locating property...',
      'Extracting satellite features...',
      'Running ML model...',
      'Applying safety rules...',
      'Generating projections...',
    ];
    
    let stageIdx = 0;
    setLoadingStage(stages[0]);
    const stageInterval = setInterval(() => {
      stageIdx = (stageIdx + 1) % stages.length;
      setLoadingStage(stages[stageIdx]);
    }, 800);
    
    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude: lat, longitude: lon }),
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze property');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(stageInterval);
      setLoading(false);
      setLoadingStage('');
    }
  };

  const handleExample = (value) => {
    setSearchMode('address');
    setAddress(value);
    setTimeout(() => handleSearch(value), 50);
  };

  return (
    <div className="min-h-screen relative overflow-x-hidden">
      <div className="gradient-orb orb-1" />
      <div className="gradient-orb orb-2" />
      <div className="gradient-orb orb-3" />
      
      <div className="relative z-10">
        <Header />
        
        <main className="max-w-7xl mx-auto px-6 pb-12">
          <SearchSection 
            searchMode={searchMode}
            setSearchMode={setSearchMode}
            address={address}
            setAddress={setAddress}
            coords={coords}
            setCoords={setCoords}
            pickedLocation={pickedLocation}
            setPickedLocation={setPickedLocation}
            onSearch={handleSearch}
            onSearchByCoords={handleSearchByCoords}
            onExample={handleExample}
            loading={loading}
            riskGrid={riskGrid}
          />
          
          <AnimatePresence>
            {loading && <LoadingState stage={loadingStage} />}
          </AnimatePresence>
          
          <AnimatePresence>
            {error && <ErrorState message={error} />}
          </AnimatePresence>
          
          <AnimatePresence>
            {result && !loading && <ResultsView data={result} riskGrid={riskGrid} />}
          </AnimatePresence>
          
          {!result && !loading && !error && <WelcomeState riskGrid={riskGrid} />}
        </main>
        
        <Footer />
      </div>
    </div>
  );
}

// ─── Header ──────────────────────────────────────────────
function Header() {
  return (
    <header className="relative z-10 px-6 py-12 md:py-20">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center glow-purple">
              <Cloud className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-xs font-bold text-violet-300 tracking-widest">FLOOD RISK INTELLIGENCE</div>
              <div className="text-sm text-white/40">Kerala Property Assessment</div>
            </div>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-black tracking-tight leading-none">
            <span className="text-white">Predict the</span>
            <br/>
            <span className="gradient-text">unpredictable.</span>
          </h1>
          
          <p className="text-white/60 text-lg md:text-xl mt-6 max-w-2xl leading-relaxed">
            AI-powered flood risk for any Kerala property. Combining satellite data, 
            climate science, and physics-based safety rules.
          </p>
          
          <div className="flex flex-wrap gap-3 mt-8">
            <Badge color="green" pulse>Live System</Badge>
            <Badge>RBI 2024 Compliant</Badge>
            <Badge>91.96% Accuracy</Badge>
            <Badge>5 Satellite Datasets</Badge>
          </div>
        </motion.div>
      </div>
    </header>
  );
}

function Badge({ children, color = 'default', pulse = false }) {
  const colors = {
    default: 'bg-white/5 text-white/80 border-white/10',
    green: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
  };
  return (
    <span className={`px-4 py-1.5 rounded-full text-xs font-medium border backdrop-blur-sm ${colors[color]}`}>
      {pulse && <span className="inline-block w-1.5 h-1.5 bg-emerald-400 rounded-full mr-2 animate-pulse"></span>}
      {children}
    </span>
  );
}


// ─── Heatmap Layer Component ──────────────────────────────
function HeatmapLayer({ points }) {
  const map = useMap();
  const heatLayerRef = useRef(null);
  
  useEffect(() => {
    if (!points || points.length === 0) return;
    
    // Format points for leaflet.heat: [lat, lng, intensity]
    const heatData = points.map(p => [p.lat, p.lng, p.risk]);
    
    // Create heat layer with custom gradient
    const heatLayer = L.heatLayer(heatData, {
      radius: 18,
      blur: 22,
      maxZoom: 12,
      max: 1.0,
      minOpacity: 0.4,
      gradient: {
        0.0: '#10b981',  // emerald (very low)
        0.25: '#84cc16', // lime (low)
        0.5: '#f59e0b',  // amber (moderate)
        0.7: '#f97316',  // orange (high)
        0.9: '#dc2626',  // red (critical)
        1.0: '#831843',  // deep red (extreme)
      }
    });
    
    heatLayer.addTo(map);
    heatLayerRef.current = heatLayer;
    
    return () => {
      if (heatLayerRef.current) {
        map.removeLayer(heatLayerRef.current);
      }
    };
  }, [points, map]);
  
  return null;
}


// ─── Autosuggest Input ──────────────────────────
function AutosuggestInput({ value, onChange, onSelect, disabled, onEnter }) {
  const [suggestions, setSuggestions] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [loading, setLoading] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const debounceTimerRef = useRef(null);
  const containerRef = useRef(null);
  
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  const fetchSuggestions = async (query) => {
    if (query.length < 2) {
      setSuggestions([]);
      setShowDropdown(false);
      return;
    }
    
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/suggest?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      const sugs = data.suggestions || [];
      setSuggestions(sugs);
      setShowDropdown(sugs.length > 0);
      setHighlightedIndex(-1);
    } catch (err) {
      setSuggestions([]);
      setShowDropdown(false);
    } finally {
      setLoading(false);
    }
  };
  
  const handleChange = (e) => {
    const newValue = e.target.value;
    onChange(newValue);
    
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    debounceTimerRef.current = setTimeout(() => {
      fetchSuggestions(newValue);
    }, 300);
  };
  
  const handleSelect = (suggestion) => {
    onChange(suggestion.display_name);
    setShowDropdown(false);
    setSuggestions([]);
    onSelect(suggestion);
  };
  
  const handleKeyDown = (e) => {
    if (!showDropdown || suggestions.length === 0) {
      if (e.key === 'Enter' && onEnter) onEnter();
      return;
    }
    
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex(i => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex(i => Math.max(i - 1, -1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightedIndex >= 0) {
        handleSelect(suggestions[highlightedIndex]);
      } else if (onEnter) {
        onEnter();
      }
    } else if (e.key === 'Escape') {
      setShowDropdown(false);
    }
  };
  
  return (
    <div ref={containerRef} className="flex-1 relative">
      <MapPin className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30 z-10" />
      <input
        type="text"
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
        placeholder="Enter a place (e.g., Kuttanad, Aluva, Munnar)"
        className="w-full pl-12 pr-12 py-4 rounded-2xl bg-white/5 border border-white/10 focus:border-violet-500/50 focus:bg-white/10 focus:outline-none transition-all text-white placeholder:text-white/30"
        disabled={disabled}
        autoComplete="off"
      />
      
      {loading && (
        <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-violet-400 animate-spin" />
      )}
      
      <AnimatePresence>
        {showDropdown && suggestions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full mt-2 w-full glass-card overflow-hidden"
            style={{ background: 'rgba(15, 15, 25, 0.98)', zIndex: 9999 }}
          >
            <div className="max-h-80 overflow-y-auto">
              {suggestions.map((sug, idx) => {
                const typeIcon = getTypeIcon(sug.type);
                return (
                  <motion.button
                    key={`${sug.latitude}-${sug.longitude}-${idx}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.04 }}
                    onClick={() => handleSelect(sug)}
                    onMouseEnter={() => setHighlightedIndex(idx)}
                    className={`w-full text-left px-4 py-3 transition-all border-b border-white/5 last:border-b-0 flex items-start gap-3 ${
                      highlightedIndex === idx 
                        ? 'bg-violet-500/20 border-violet-500/30' 
                        : 'hover:bg-white/5'
                    }`}
                  >
                    <div className="text-xl mt-0.5 shrink-0">{typeIcon}</div>
                    <div className="flex-1 min-w-0">
                      <div className="text-white font-medium text-sm truncate">
                        {sug.display_name}
                      </div>
                      <div className="text-white/40 text-xs truncate mt-0.5">
                        {sug.full_name.split(',').slice(2, 5).join(',')}
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-white/30 mt-1 shrink-0" />
                  </motion.button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}


// ─── Search Section ──────────────────────────────────────
function SearchSection({ searchMode, setSearchMode, address, setAddress, coords, setCoords, pickedLocation, setPickedLocation, onSearch, onSearchByCoords, onExample, loading, riskGrid }) {
  const handleSuggestionSelect = (sug) => {
    onSearchByCoords(sug.latitude, sug.longitude);
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="glass-card p-6 md:p-8 mb-8 relative"
      style={{ zIndex: 100 }}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Search className="w-5 h-5 text-violet-400" />
          <h2 className="font-semibold text-white">Search Property</h2>
        </div>
        
        <div className="flex bg-white/5 rounded-2xl p-1 border border-white/10">
          {[
            { id: 'address', icon: Search, label: 'Address' },
            { id: 'coords', icon: Crosshair, label: 'Coordinates' },
            { id: 'map', icon: MapIcon, label: 'Pick on Map' },
          ].map((mode) => {
            const Icon = mode.icon;
            const isActive = searchMode === mode.id;
            return (
              <button
                key={mode.id}
                onClick={() => setSearchMode(mode.id)}
                className={`relative px-3 md:px-4 py-2 rounded-xl text-xs md:text-sm font-medium transition-all flex items-center gap-2 ${
                  isActive ? 'text-white' : 'text-white/50 hover:text-white/80'
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeMode"
                    className="absolute inset-0 bg-gradient-to-r from-violet-500 to-blue-500 rounded-xl"
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                <span className="relative z-10 flex items-center gap-2">
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{mode.label}</span>
                </span>
              </button>
            );
          })}
        </div>
      </div>
      
      <AnimatePresence mode="wait">
        {searchMode === 'address' && (
          <motion.div
            key="address"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <div className="flex flex-col md:flex-row gap-3">
              <AutosuggestInput
                value={address}
                onChange={setAddress}
                onSelect={handleSuggestionSelect}
                onEnter={() => onSearch()}
                disabled={loading}
              />
              <SearchButton onClick={() => onSearch()} loading={loading} disabled={!address.trim()} />
            </div>
            
            <div className="flex flex-wrap gap-2 mt-4">
              <span className="text-xs text-white/40 font-medium pt-1.5">Try:</span>
              {EXAMPLE_ADDRESSES.map((ex) => (
                <button
                  key={ex.value}
                  onClick={() => onExample(ex.value)}
                  disabled={loading}
                  className="px-3 py-1.5 rounded-full text-xs font-medium bg-white/5 hover:bg-white/10 text-white/70 hover:text-white transition-all border border-white/10 disabled:opacity-50"
                >
                  {ex.emoji} {ex.label}
                </button>
              ))}
            </div>
          </motion.div>
        )}
        
        {searchMode === 'coords' && (
          <motion.div
            key="coords"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <div className="grid md:grid-cols-3 gap-3">
              <div>
                <label className="text-xs text-white/50 mb-1.5 block">Latitude (8.0 - 12.8)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={coords.lat}
                  onChange={(e) => setCoords({ ...coords, lat: e.target.value })}
                  placeholder="9.4507"
                  className="w-full px-4 py-3 rounded-2xl bg-white/5 border border-white/10 focus:border-violet-500/50 focus:bg-white/10 focus:outline-none text-white"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 mb-1.5 block">Longitude (74.5 - 77.5)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={coords.lon}
                  onChange={(e) => setCoords({ ...coords, lon: e.target.value })}
                  placeholder="76.4308"
                  className="w-full px-4 py-3 rounded-2xl bg-white/5 border border-white/10 focus:border-violet-500/50 focus:bg-white/10 focus:outline-none text-white"
                />
              </div>
              <div className="flex items-end">
                <SearchButton onClick={() => onSearch()} loading={loading} fullWidth />
              </div>
            </div>
            <p className="text-xs text-white/40 mt-3">
              💡 Get exact coordinates from Google Maps by right-clicking any location
            </p>
          </motion.div>
        )}
        
        {searchMode === 'map' && (
          <motion.div
            key="map"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <div className="rounded-2xl overflow-hidden border border-white/10 mb-4" style={{ height: '400px' }}>
              <PickerMap pickedLocation={pickedLocation} setPickedLocation={setPickedLocation} riskGrid={riskGrid} />
            </div>
            
            <div className="flex flex-col md:flex-row gap-3 items-center">
              <div className="flex-1 text-sm text-white/60">
                {pickedLocation ? (
                  <span>📍 Selected: <strong className="text-white">{pickedLocation.lat.toFixed(4)}, {pickedLocation.lng.toFixed(4)}</strong></span>
                ) : (
                  <span>Click anywhere on the map to pick a location</span>
                )}
              </div>
              <SearchButton onClick={() => onSearch()} loading={loading} disabled={!pickedLocation} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function SearchButton({ onClick, loading, disabled, fullWidth }) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      disabled={loading || disabled}
      className={`${fullWidth ? 'w-full' : ''} px-8 py-4 rounded-2xl bg-gradient-to-r from-violet-600 to-blue-600 hover:from-violet-500 hover:to-blue-500 text-white font-semibold shadow-lg shadow-violet-500/30 hover:shadow-violet-500/50 transition-all disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-2`}
    >
      {loading ? (
        <Loader2 className="w-5 h-5 animate-spin" />
      ) : (
        <>
          Analyze
          <ArrowUpRight className="w-5 h-5" />
        </>
      )}
    </motion.button>
  );
}

// ─── Picker Map Component ────────────────────────────────
function PickerMap({ pickedLocation, setPickedLocation, riskGrid }) {
  const center = pickedLocation || { lat: 10.0, lng: 76.3 };
  const [showHeatmap, setShowHeatmap] = useState(true);
  
  function MapClickHandler() {
    useMapEvents({
      click(e) {
        const { lat, lng } = e.latlng;
        if (lat >= 8.0 && lat <= 12.8 && lng >= 74.5 && lng <= 77.5) {
          setPickedLocation({ lat, lng });
        }
      },
    });
    return null;
  }
  
  return (
    <div className="relative w-full h-full">
      <MapContainer 
        center={[center.lat, center.lng]} 
        zoom={9} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='Esri'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />
        <TileLayer
          attribution='Esri Labels'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />
        <MapClickHandler />
        {showHeatmap && riskGrid && riskGrid.length > 0 && (
          <HeatmapLayer points={riskGrid} />
        )}
        {pickedLocation && (
          <Marker position={[pickedLocation.lat, pickedLocation.lng]} icon={PIN_ICON} />
        )}
      </MapContainer>
      
      {/* Heatmap toggle button */}
      {riskGrid && riskGrid.length > 0 && (
        <HeatmapToggle showHeatmap={showHeatmap} setShowHeatmap={setShowHeatmap} />
      )}
      
      {/* Heatmap legend */}
      {showHeatmap && riskGrid && riskGrid.length > 0 && <HeatmapLegend />}
    </div>
  );
}

// ─── Heatmap Toggle Button ────────────────────────────────
function HeatmapToggle({ showHeatmap, setShowHeatmap }) {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => setShowHeatmap(!showHeatmap)}
      className={`absolute top-4 right-4 px-4 py-2.5 rounded-xl flex items-center gap-2 text-sm font-semibold backdrop-blur-md transition-all ${
        showHeatmap
          ? 'bg-gradient-to-r from-red-500/90 to-orange-500/90 text-white border-2 border-white/20 shadow-lg shadow-red-500/30'
          : 'bg-white/10 text-white/80 border-2 border-white/10 hover:bg-white/20'
      }`}
      style={{ zIndex: 1000 }}
    >
      {showHeatmap ? (
        <>
          <Flame className="w-4 h-4" />
          <span>Heatmap ON</span>
        </>
      ) : (
        <>
          <EyeOff className="w-4 h-4" />
          <span>Show Risk Heatmap</span>
        </>
      )}
    </motion.button>
  );
}

// ─── Heatmap Legend ───────────────────────────────────────
function HeatmapLegend() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="absolute bottom-4 left-4 glass-card-light p-3 backdrop-blur-md"
      style={{ zIndex: 1000, background: 'rgba(15, 15, 25, 0.85)' }}
    >
      <div className="text-xs font-bold text-white/80 mb-2 tracking-wider">FLOOD RISK</div>
      <div className="flex items-center gap-2">
        <div className="text-xs text-white/60">Low</div>
        <div 
          className="w-32 h-2 rounded-full" 
          style={{
            background: 'linear-gradient(90deg, #10b981 0%, #84cc16 25%, #f59e0b 50%, #f97316 70%, #dc2626 90%, #831843 100%)'
          }}
        />
        <div className="text-xs text-white/60">High</div>
      </div>
    </motion.div>
  );
}

// ─── Loading State ───────────────────────────────────────
function LoadingState({ stage }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="glass-card p-12 text-center"
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        className="w-16 h-16 mx-auto mb-6 rounded-full border-4 border-violet-500/20 border-t-violet-500"
      />
      <p className="text-lg font-medium text-white">{stage}</p>
      <p className="text-sm text-white/40 mt-2">Analyzing satellite data...</p>
    </motion.div>
  );
}

// ─── Error State ─────────────────────────────────────────
function ErrorState({ message }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="glass-card p-6 flex items-start gap-4 border-red-500/30"
      style={{ background: 'rgba(239, 68, 68, 0.1)' }}
    >
      <AlertTriangle className="w-6 h-6 text-red-400 shrink-0 mt-0.5" />
      <div>
        <h3 className="font-semibold text-red-300">Could not analyze property</h3>
        <p className="text-red-200/80 text-sm mt-1">{message}</p>
      </div>
    </motion.div>
  );
}

// ─── Welcome State ───────────────────────────────────────
function WelcomeState({ riskGrid }) {
  return (
    <>
      {/* Kerala-wide heatmap preview */}
      {riskGrid && riskGrid.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card overflow-hidden mb-6 relative"
          style={{ height: '400px' }}
        >
          <KeralaPreviewMap riskGrid={riskGrid} />
          <div className="absolute top-4 left-4 glass-card-light px-4 py-2 backdrop-blur-md" style={{ zIndex: 1000, background: 'rgba(15, 15, 25, 0.85)' }}>
            <div className="text-xs text-white/60">EXPLORE</div>
            <div className="text-sm font-bold text-white flex items-center gap-2">
              <Flame className="w-4 h-4 text-red-400" />
              Kerala Flood Risk Heatmap
            </div>
          </div>
        </motion.div>
      )}
      
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="grid md:grid-cols-2 gap-6"
      >
        <div className="glass-card p-8">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-500/20 to-blue-500/20 border border-violet-500/30 flex items-center justify-center mb-6">
            <Sparkles className="w-6 h-6 text-violet-400" />
          </div>
          <h3 className="text-2xl font-bold text-white mb-6">How it works</h3>
          <div className="space-y-5">
            {[
              { num: 1, title: 'Geocode address', desc: 'GPS coordinates from any Kerala address' },
              { num: 2, title: 'Extract features', desc: 'Elevation, slope, drainage, river distance' },
              { num: 3, title: 'Hybrid AI scoring', desc: 'ML model + 12 physics-based rules' },
              { num: 4, title: 'Climate projection', desc: 'Project to 2030 & 2040 using IPCC' },
            ].map((step) => (
              <div key={step.num} className="flex gap-4">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-blue-500 text-white font-bold flex items-center justify-center shrink-0 text-sm">
                  {step.num}
                </div>
                <div>
                  <h4 className="font-semibold text-white">{step.title}</h4>
                  <p className="text-sm text-white/60">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="glass-card p-8">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30 flex items-center justify-center mb-6">
            <Activity className="w-6 h-6 text-emerald-400" />
          </div>
          <h3 className="text-2xl font-bold text-white mb-6">By the numbers</h3>
          <div className="grid grid-cols-2 gap-4">
            {[
              { num: '20,481', label: 'Houses destroyed in 2018' },
              { num: '11,873', label: 'Rivers analyzed' },
              { num: '±2m', label: 'Elevation accuracy' },
              { num: '91.96%', label: 'Model accuracy' },
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 + i * 0.1 }}
                className="glass-card-light p-5"
              >
                <div className="text-3xl font-black text-white">{stat.num}</div>
                <div className="text-xs text-white/50 mt-1">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
    </>
  );
}

// ─── Kerala Preview Map (welcome screen) ──────────────────
function KeralaPreviewMap({ riskGrid }) {
  return (
    <div className="relative w-full h-full">
      <MapContainer 
        center={[10.4, 76.2]} 
        zoom={7} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='Esri'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />
        <TileLayer
          attribution='Esri Labels'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />
        <HeatmapLayer points={riskGrid} />
      </MapContainer>
      <HeatmapLegend />
    </div>
  );
}

// ─── Results View ────────────────────────────────────────
function ResultsView({ data, riskGrid }) {
  const severity = SEVERITY_STYLES[data.risk.severity] || SEVERITY_STYLES['MODERATE'];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6 relative"
      style={{ zIndex: 1 }}
    >
      <RiskHeroCard data={data} severity={severity} />
      
      {data.risk.used_rule_override && (
        <RuleOverrideBanner adjustments={data.risk.rule_adjustments} mlPct={data.risk.ml_percentage} finalPct={data.risk.final_percentage} />
      )}
      
      <div className="grid md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <MapView data={data} severity={severity} riskGrid={riskGrid} />
        </div>
        <ClimateProjections projections={data.climate_projections} />
      </div>
      
      <RecommendationCard recommendation={data.recommendation} />
      <PropertyFeatures features={data.features} />
      <FinancialImpact finances={data.financial_impact} />
      {data.shap_explanation && data.shap_explanation.length > 0 && (
        <ShapChart shap={data.shap_explanation} />
      )}
    </motion.div>
  );
}

// ─── Risk Hero Card ──────────────────────────────────────
function RiskHeroCard({ data, severity }) {
  const pct = data.risk.final_percentage;
  
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={`glass-card p-10 relative overflow-hidden ${severity.glow}`}
    >
      <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${severity.gradient}`}></div>
      <div className={`absolute -top-32 -right-32 w-64 h-64 bg-gradient-to-br ${severity.gradient} opacity-20 blur-3xl rounded-full`}></div>
      
      <div className="relative z-10">
        <div className="text-xs font-bold tracking-widest text-white/40 uppercase mb-3">
          Flood Risk Assessment
        </div>
        
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', delay: 0.2, stiffness: 100 }}
          className={`text-7xl md:text-9xl font-black bg-gradient-to-br ${severity.gradient} bg-clip-text text-transparent leading-none my-2`}
          style={{ letterSpacing: '-0.05em' }}
        >
          {pct}%
        </motion.div>
        
        <div className={`text-2xl font-bold ${severity.text} mt-4`}>
          {data.risk.emoji} {data.risk.severity}
        </div>
        
        <div className="mt-6 flex items-center gap-2 text-sm text-white/50">
          <MapPin className="w-4 h-4" />
          {data.coordinates.latitude.toFixed(4)}, {data.coordinates.longitude.toFixed(4)}
        </div>
      </div>
    </motion.div>
  );
}

// ─── Rule Override Banner ────────────────────────────────
function RuleOverrideBanner({ adjustments, mlPct, finalPct }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.3 }}
      className="glass-card p-5"
      style={{ background: 'linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.05))', borderColor: 'rgba(245,158,11,0.3)' }}
    >
      <div className="flex gap-4">
        <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center shrink-0">
          <Shield className="w-5 h-5 text-amber-400" />
        </div>
        <div className="flex-1">
          <h4 className="font-bold text-amber-300 mb-1">Hybrid Decision Applied</h4>
          <p className="text-sm text-amber-200/80">
            ML model predicted <strong className="text-amber-200">{mlPct}%</strong>. Safety rule adjusted this to <strong className="text-amber-200">{finalPct}%</strong>:
            <em className="block mt-1 text-amber-200/60">{adjustments[0]?.reason}</em>
          </p>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Map View ────────────────────────────────────────────
function MapView({ data, severity, riskGrid }) {
  const position = [data.coordinates.latitude, data.coordinates.longitude];
  const customIcon = createCustomIcon(severity.accent);
  const [showHeatmap, setShowHeatmap] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.4 }}
      className="glass-card overflow-hidden h-[450px] relative"
    >
      <MapContainer 
        center={position} 
        zoom={15} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='Esri'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />
        <TileLayer
          attribution='Esri Labels'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />
        
        {showHeatmap && riskGrid && riskGrid.length > 0 && (
          <HeatmapLayer points={riskGrid} />
        )}
        
        <Circle 
          center={position} 
          radius={200} 
          pathOptions={{ 
            color: severity.accent, 
            fillColor: severity.accent, 
            fillOpacity: 0.15,
            weight: 2
          }} 
        />
        
        <Marker position={position} icon={customIcon}>
          <Popup>
            <div style={{ textAlign: 'center', padding: '8px', color: 'white' }}>
              <div style={{ color: severity.accent, fontWeight: 'bold', fontSize: '14px' }}>
                {data.risk.emoji} {data.risk.severity} RISK
              </div>
              <div style={{ fontSize: '24px', fontWeight: '900', margin: '4px 0', color: 'white' }}>
                {data.risk.final_percentage}%
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)' }}>
                {data.coordinates.latitude.toFixed(4)}, {data.coordinates.longitude.toFixed(4)}
              </div>
            </div>
          </Popup>
        </Marker>
      </MapContainer>
      
      {riskGrid && riskGrid.length > 0 && (
        <HeatmapToggle showHeatmap={showHeatmap} setShowHeatmap={setShowHeatmap} />
      )}
      
      {showHeatmap && <HeatmapLegend />}
    </motion.div>
  );
}

// ─── Climate Projections ─────────────────────────────────
function ClimateProjections({ projections }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.4 }}
      className="glass-card p-6"
    >
      <div className="flex items-center gap-2 mb-1">
        <TrendingUp className="w-5 h-5 text-violet-400" />
        <h3 className="font-bold text-white">Climate Timeline</h3>
      </div>
      <p className="text-xs text-white/40 mb-4">IPCC RCP 4.5 projections</p>
      
      <div className="space-y-3">
        {Object.entries(projections).map(([year, data], i) => {
          const yearLabel = year === 'today' ? 'Today' : year;
          const sev = SEVERITY_STYLES[data.severity] || SEVERITY_STYLES['MODERATE'];
          
          return (
            <motion.div
              key={year}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + i * 0.15 }}
              className="glass-card-light p-4"
              style={{ borderLeft: `3px solid ${sev.accent}` }}
            >
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold text-white/80">{yearLabel}</span>
                <span className={`font-black text-lg ${sev.text}`}>{data.probability}%</span>
              </div>
              <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${data.probability}%` }}
                  transition={{ delay: 0.7 + i * 0.15, duration: 1 }}
                  className="h-full rounded-full"
                  style={{ background: `linear-gradient(90deg, ${sev.accent}, ${sev.accent}cc)` }}
                />
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

// ─── Recommendation ──────────────────────────────────────
function RecommendationCard({ recommendation }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.6 }}
      className="glass-card p-6"
      style={{ background: 'linear-gradient(135deg, rgba(139,92,246,0.08), rgba(59,130,246,0.05))' }}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <h3 className="font-bold text-white">Recommendation</h3>
      </div>
      <p className="text-white/80">{recommendation}</p>
    </motion.div>
  );
}

// ─── Property Features ───────────────────────────────────
function PropertyFeatures({ features }) {
  const items = [
    { icon: Mountain, label: 'Elevation', value: `${features.elevation_m}m`, hint: 'Above sea level', color: 'text-violet-400' },
    { icon: Droplets, label: 'Drainage', value: `${features.upstream_catchment_km2}km²`, hint: 'Upstream area', color: 'text-blue-400' },
    { icon: Compass, label: 'Slope', value: `${features.slope_degrees}°`, hint: 'Land steepness', color: 'text-cyan-400' },
    { 
      icon: Activity, 
      label: 'Nearest River', 
      value: features.distance_to_river_m < 30 
        ? 'Adjacent' 
        : features.distance_to_river_m < 1000 
          ? `${features.distance_to_river_m}m` 
          : `${(features.distance_to_river_m/1000).toFixed(1)}km`, 
      hint: features.distance_to_river_m < 30 ? '⚠️ On waterway' : 'Distance', 
      color: features.distance_to_river_m < 30 ? 'text-red-400' : 'text-emerald-400' 
    },
    { icon: TreePine, label: 'Reclaimed Paddy', value: features.is_historical_paddy ? 'Yes' : 'No', hint: 'Land history', color: features.is_historical_paddy ? 'text-amber-400' : 'text-emerald-400' },
  ];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.7 }}
      className="glass-card p-6"
    >
      <h3 className="font-bold text-white mb-5">Property Characteristics</h3>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {items.map((item, i) => {
          const Icon = item.icon;
          return (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.1 }}
              className="glass-card-light p-4 hover:bg-white/10 transition-all"
            >
              <Icon className={`w-5 h-5 ${item.color} mb-2`} />
              <div className="text-xs text-white/50 uppercase tracking-wider font-medium">{item.label}</div>
              <div className="text-xl font-bold text-white mt-1">{item.value}</div>
              <div className="text-xs text-white/30 mt-1">{item.hint}</div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

// ─── Financial Impact ────────────────────────────────────
function FinancialImpact({ finances }) {
  const items = [
    { label: 'Expected Loss', value: finances.rebuilding_loss_str, gradient: 'from-red-400 to-pink-500' },
    { label: 'Insurance Extra', value: finances.insurance_str, gradient: 'from-orange-400 to-red-500' },
    { label: 'Resale Risk', value: finances.resale_loss_str, gradient: 'from-amber-400 to-orange-500' },
    { label: 'Total Impact', value: finances.total_5yr_str, gradient: 'from-violet-400 to-pink-500' },
  ];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.9 }}
      className="glass-card p-6"
    >
      <div className="flex items-center gap-2 mb-5">
        <IndianRupee className="w-5 h-5 text-emerald-400" />
        <h3 className="font-bold text-white">Financial Impact (5-year estimate)</h3>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {items.map((item, i) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1 + i * 0.1 }}
            className="glass-card-light p-5"
          >
            <div className="text-xs text-white/50 uppercase tracking-wider font-medium mb-2">{item.label}</div>
            <div className={`text-2xl font-bold bg-gradient-to-r ${item.gradient} bg-clip-text text-transparent`}>
              {item.value}
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
// ─── SHAP Explanation Chart ───────────────────────────────
function ShapChart({ shap }) {
  const FEATURE_LABELS = {
    elevation_m:              { label: 'Elevation',       unit: 'm',   icon: '⛰️' },
    slope_degrees:            { label: 'Slope',           unit: '°',   icon: '📐' },
    distance_to_river_m:      { label: 'River Distance',  unit: 'm',   icon: '🌊' },
    upstream_catchment_km2:   { label: 'Drainage Area',   unit: 'km²', icon: '💧' },
    is_historical_paddy:      { label: 'Paddy Land',      unit: '',    icon: '🌾' },
  };

  // Sort by absolute SHAP value descending
  const sorted = [...shap].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

  // Find max absolute value for bar scaling
  const maxAbs = Math.max(...sorted.map(s => Math.abs(s.shap_value)), 0.01);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1.0 }}
      className="glass-card p-6"
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-2">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
          <span className="text-white text-sm font-bold">∑</span>
        </div>
        <div>
          <h3 className="font-bold text-white">Why This Score?</h3>
          <p className="text-xs text-white/40">SHAP feature contributions — what drove the ML prediction</p>
        </div>
      </div>

      <p className="text-xs text-white/50 mb-5">
        Each bar shows how much that terrain feature pushed the flood risk score up (red) or down (blue) from Kerala's average risk.
      </p>

      {/* Bars */}
      <div className="space-y-3">
        {sorted.map((item, i) => {
          const meta  = FEATURE_LABELS[item.feature] || { label: item.feature, unit: '', icon: '📊' };
          const pct   = (Math.abs(item.shap_value) / maxAbs) * 100;
          const isPos = item.shap_value >= 0;
          const sign  = isPos ? '+' : '−';
          const shapPct = (Math.abs(item.shap_value) * 100).toFixed(1);

          // Format raw value
          let rawVal = '';
          if (item.feature === 'is_historical_paddy') {
            rawVal = item.value ? 'Yes' : 'No';
          } else if (item.feature === 'distance_to_river_m' && item.value >= 1000) {
            rawVal = `${(item.value / 1000).toFixed(1)}km`;
          } else {
            rawVal = `${Number(item.value).toFixed(1)}${meta.unit}`;
          }

          return (
            <motion.div
              key={item.feature}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.1 + i * 0.08 }}
              className="glass-card-light p-3 rounded-2xl"
            >
              {/* Feature label row */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{meta.icon}</span>
                  <div>
                    <span className="text-sm font-semibold text-white">{meta.label}</span>
                    <span className="text-xs text-white/40 ml-2">({rawVal})</span>
                  </div>
                </div>
                <span
                  className="text-sm font-bold"
                  style={{ color: isPos ? '#f97316' : '#60a5fa' }}
                >
                  {sign}{shapPct}%
                </span>
              </div>

              {/* Bar track */}
              <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ delay: 1.2 + i * 0.08, duration: 0.8, ease: 'easeOut' }}
                  className="absolute top-0 h-full rounded-full"
                  style={{
                    background: isPos
                      ? 'linear-gradient(90deg, #f97316, #dc2626)'
                      : 'linear-gradient(90deg, #3b82f6, #06b6d4)',
                    left: 0,
                  }}
                />
              </div>

              {/* Contribution label */}
              <div className="mt-1">
                <span
                  className="text-xs"
                  style={{ color: isPos ? '#f97316aa' : '#60a5faaa' }}
                >
                  {isPos ? '▲ increases flood risk' : item.shap_value === 0 ? '● no contribution' : '▼ decreases flood risk'}
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Footer note */}
      <div className="mt-4 pt-4 border-t border-white/10">
        <p className="text-xs text-white/30 text-center">
          SHAP (SHapley Additive exPlanations) — mathematically rigorous feature attribution.
          Values show contribution to ML score before hybrid rules are applied.
        </p>
      </div>
    </motion.div>
  );
}
// ─── Footer ──────────────────────────────────────────────
function Footer() {
  return (
    <footer className="border-t border-white/5 mt-16 py-8">
      <div className="max-w-7xl mx-auto px-6 text-center">
        <p className="text-sm text-white/40">Hybrid ML + Rule-Based Flood Risk System</p>
        <p className="text-xs mt-2 text-white/30">Random Forest 91.96% accuracy · 12 physics rules · RBI 2024 Compliant</p>
      </div>
    </footer>
  );
}