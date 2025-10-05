document.addEventListener('DOMContentLoaded', () => {

  

  // ======================================================================
  // 2. Main Map Initialization and Utility Functions
  // ======================================================================

  function safeAddListener(el, ev, handler, opts) {
    if (!el) {
      console.warn(`Element for event ${ev} is missing`);
      return;
    }
    el.addEventListener(ev, handler, opts);
  }

  // Initialize map: start NA, enable full world pan/zoom
  const map = L.map('map', {
    center: [50, -100],
    zoom: 4,
    minZoom: 2,
    maxZoom: 9,
    zoomControl: true
  });

  const mapboxToken = "pk.eyJ1IjoidGFzY2lvb28iLCJhIjoiY21nMHNhaDIyMGh6MzJpcXdya2s1MzB2cSJ9.-nmfK81AhMLk9cgmu-7QrA";

  const satelliteStreets = L.tileLayer(
    `https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/{z}/{x}/{y}?access_token=${mapboxToken}`,
    { tileSize: 512, zoomOffset: -1, attribution: 'Imagery © Mapbox, © Maxar, © Esri | © OSM' }
  );

  const streets = L.tileLayer(
    `https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/{z}/{x}/{y}?access_token=${mapboxToken}`,
    { tileSize: 512, zoomOffset: -1, attribution: 'Map data © OSM contributors' }
  );

  satelliteStreets.addTo(map);
  let currentBaseLayer = "satellite";

  map.attributionControl.remove();


let weatherWidget = document.getElementById('weather-widget');
if (!weatherWidget) {
  weatherWidget = document.createElement('div');
  weatherWidget.id = 'weather-widget';
  weatherWidget.style.position = 'absolute';
  weatherWidget.style.left = '10px';
  weatherWidget.style.top = '170px';
  weatherWidget.style.width = '410px';
  weatherWidget.style.background = 'rgba(249,251,254,0.87)';
  weatherWidget.style.boxShadow = '0 2px 12px rgba(0,0,0,0.11)';
  weatherWidget.style.borderRadius = '10px';
  weatherWidget.style.padding = '12px 12px 16px 12px';
  weatherWidget.style.zIndex = '1010';
  weatherWidget.style.fontFamily = 'inherit';
  weatherWidget.style.transition = 'background .3s, color .3s';
  document.body.appendChild(weatherWidget);
}

weatherWidget.innerHTML = `
  <div id="weather-title" style="font-weight:500;font-size:1.1em;margin-bottom:8px;">
    Weather Forecast · New York, NY
  </div>
  <div id="weather-selected-city" style="font-size:1em;margin-bottom:7px;">
    New York, NY, USA
  </div>
  <div id="weather-forecast-week" style="display:flex;gap:13px;margin-bottom:7px;"></div>
  <div id="weather-aqi-row" style="display:flex;gap:18px;justify-content:flex-start;margin-bottom:4px;margin-top:4px;"></div>
  <div id="weather-status" style="margin-top:4px;color:#a00;font-size:.95em;"></div>
`;

function updateWeatherWidgetTheme() {
  const dark = document.body.classList.contains('dark-mode');
  weatherWidget.style.background = dark ? "#171c24" : "#f9fbfe";
  weatherWidget.style.color = dark ? "#e3e6eb" : "#232426";
  weatherWidget.style.boxShadow = dark ? "0 2px 13px rgba(0,0,0,0.65)" : "0 2px 12px rgba(0,0,0,0.11)";
  // Extra: toggle link, border, etc if you like
}

updateWeatherWidgetTheme();
if (window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.body.classList.add('dark-mode');
document.body.addEventListener('click', updateWeatherWidgetTheme);
window.addEventListener('DOMContentLoaded', updateWeatherWidgetTheme);

// You can also attach to your existing dark mode toggle:
const darkModeToggle = document.getElementById('dark-mode-toggle');
if (darkModeToggle) darkModeToggle.addEventListener('click', updateWeatherWidgetTheme);

function getWeatherEmoji(shortForecast) {
  const fc = shortForecast.toLowerCase();
  if (fc.includes("cloudy")) return "☁️";
  if (fc.includes("sunny") || fc.includes("clear")) return "☀️";
  if (fc.includes("rain") || fc.includes("showers")) return "🌧️";
  if (fc.includes("thunder")) return "⛈️";
  if (fc.includes("snow")) return "❄️";
  if (fc.includes("fog")) return "🌫️";
  if (fc.includes("wind")) return "🌬️";
  return "🌡️";
}

function getAQIColor(aqi) {
  if (aqi == null) return '#999999';
  if (aqi <= 50) return '#00e400';
  if (aqi <= 100) return '#ffff00';
  if (aqi <= 150) return '#ff7e00';
  if (aqi <= 200) return '#ff0000';
  if (aqi <= 300) return '#99004c';
  return '#7e0023';
}

// Widget element references
const weatherSelectedCity = weatherWidget.querySelector('#weather-selected-city');
const weatherForecastWeek = weatherWidget.querySelector('#weather-forecast-week');
const weatherAqiRow = weatherWidget.querySelector('#weather-aqi-row');
const weatherStatus = weatherWidget.querySelector('#weather-status');

async function selectWeatherCity(city) {
  weatherSelectedCity.textContent = city.display_name || "New York, NY, USA";
  weatherForecastWeek.innerHTML = '';
  weatherAqiRow.innerHTML = '';
  weatherStatus.textContent = '';
  try {
    const pointsRes = await fetch(`https://api.weather.gov/points/${city.lat},${city.lon}`, {
      headers: { 'User-Agent': 'aq-weather-map-app (contact@email.com)' }
    });
    if (!pointsRes.ok) throw new Error(`NWS API points failed with ${pointsRes.status}`);
    const pointsData = await pointsRes.json();

    if (!pointsData.properties || !pointsData.properties.forecast) throw new Error('No forecast URL available');
    const forecastUrl = pointsData.properties.forecast;

    const forecastRes = await fetch(forecastUrl, {
      headers: { 'User-Agent': 'aq-weather-map-app (contact@email.com)' }
    });
    if (!forecastRes.ok) throw new Error(`NWS forecast failed with ${forecastRes.status}`);
    const forecastData = await forecastRes.json();

    if (!forecastData.properties || !forecastData.properties.periods) throw new Error('Invalid forecast data');

    // RANDOM AQI DATA: for demo only
    // Inside try block after fetching forecastData
    const geojsonUrl = `http://localhost:5001/predict/aqi_week.geojson?bbox=-74.259090,40.477399,-73.700272,40.917577&auto_step=1&target_cells=2400&min_aqi=0`;
    const geojsonResp = await fetch(geojsonUrl);
    if (!geojsonResp.ok) throw new Error(`AQI Forecast API failed: ${geojsonResp.status}`);
    const geojsonData = await geojsonResp.json();

    // Extract AQI prediction values from first feature (or adapt logic as needed)
    // Estrai tutte le feature
    const features = geojsonData.features;
    if (!features || features.length === 0) {
      throw new Error('No AQI forecast features found');
    }

    // Inizializza somma giorno per giorno
    const sumDays = [0, 0, 0, 0, 0, 0, 0];
    const featureCount = features.length;

    // Somma AQI di ciascun giorno da tutte le feature
    features.forEach(feature => {
      const p = feature.properties;
      sumDays[0] += p.AQI_pred_day1;
      sumDays[1] += p.AQI_pred_day2;
      sumDays[2] += p.AQI_pred_day3;
      sumDays[3] += p.AQI_pred_day4;
      sumDays[4] += p.AQI_pred_day5;
      sumDays[5] += p.AQI_pred_day6;
      sumDays[6] += p.AQI_pred_day7;
    });

    // Calcola le medie giorno per giorno
    const averageDays = sumDays.map(sum => (sum / featureCount).toFixed(2));

    // Passa la media al renderWeatherForecast
    renderWeatherForecast(forecastData.properties.periods, averageDays);



    weatherStatus.textContent = '';
  } catch(e) {
    weatherForecastWeek.innerHTML = '';
    weatherAqiRow.innerHTML = '';
    weatherStatus.textContent = 'Error fetching forecast: ' + e.message;
  }
}

function renderWeatherForecast(periods, aqiList) {
  function getWeekdayAbbr(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { weekday: 'short' });
  }

  weatherForecastWeek.innerHTML = '';
  weatherAqiRow.innerHTML = '';
  let shown = 0;
  for (let i = 0; i < periods.length && shown < 7; ++i) {
    const p = periods[i];
    if (p.isDaytime) {
      const dayName = getWeekdayAbbr(p.startTime);
      const emoji = getWeatherEmoji(p.shortForecast);

      // Weather icon column
      const div = document.createElement('div');
      div.style.flex = "1";
      div.style.textAlign = "center";
      div.style.minWidth = "31px";
      div.style.fontSize = "1.44em";
      div.innerHTML = `
        <div style="font-weight:600;font-size:.98em;margin-bottom:2px;">${dayName}</div>
        <div>${emoji}</div>
      `;
      weatherForecastWeek.appendChild(div);

      // AQI color dot column
      const dotAqi = document.createElement('div');
      const aqiVal = Array.isArray(aqiList) ? aqiList[shown] : null;
      dotAqi.title = `AQI: ${aqiVal}`;
      dotAqi.style.width = '16px';
      dotAqi.style.height = '16px';
      dotAqi.style.margin = 'auto';
      dotAqi.style.borderRadius = '50%';
      dotAqi.style.background = getAQIColor(aqiVal);
      dotAqi.style.border = '1.3px solid #2223';
      dotAqi.style.boxShadow = '0 0 2px #0001';
      dotAqi.style.display = 'inline-block';
      dotAqi.style.marginTop = '1px';
      dotAqi.style.marginBottom = '0px';
      dotAqi.innerHTML = `<span style="display:block;font-size:0.9em;text-align:center;color:#333;">&nbsp;</span>`;
      // Put aqi number just below the dot
      const belowNum = document.createElement('div');
      belowNum.style.fontSize = '0.91em';
      belowNum.style.color = getAQIColor(aqiVal);
      belowNum.style.marginTop = '1px';
      belowNum.style.textAlign = 'center';
      belowNum.textContent = aqiVal;
      dotAqi.appendChild(belowNum);

      weatherAqiRow.appendChild(dotAqi);

      shown++;
    }
  }
  if (shown === 0) {
    weatherForecastWeek.innerHTML = '<span style="color:#a00;">No forecast available</span>';
    weatherAqiRow.innerHTML = '';
  }
}

// Show New York forecast on load
selectWeatherCity({
  lat: 40.7128,
  lon: -74.0060,
  display_name: "New York, NY, USA"
});

selectWeatherCity({
  lat: 40.7128,
  lon: -74.0060,
  display_name: "New York, NY, USA"
});


  const streetsButton = document.getElementById('streets-button');
  const satelliteButton = document.getElementById('satellite-button');

  safeAddListener(streetsButton, 'click', () => {
    if (currentBaseLayer !== "streets") {
      if (map.hasLayer(satelliteStreets)) map.removeLayer(satelliteStreets);
      streets.addTo(map);
      currentBaseLayer = "streets";
      streetsButton && streetsButton.classList.add('active');
      satelliteButton && satelliteButton.classList.remove('active');
    }
  });

  safeAddListener(satelliteButton, 'click', () => {
    if (currentBaseLayer !== "satellite") {
      if (map.hasLayer(streets)) map.removeLayer(streets);
      satelliteStreets.addTo(map);
      currentBaseLayer = "satellite";
      satelliteButton && satelliteButton.classList.add('active');
      streetsButton && streetsButton.classList.remove('active');
    }
  });

  const fireCountDiv = document.getElementById('fireCountDiv');

  const wildfiresLayer = L.layerGroup();
  const wildfiresToggleBtn = document.getElementById('wildfires-toggle');
  let wildfiresVisible = false;

  safeAddListener(wildfiresToggleBtn, 'click', () => {
    wildfiresVisible = !wildfiresVisible;
    if (wildfiresVisible) {
      map.addLayer(wildfiresLayer);
      wildfiresToggleBtn.classList.add('active');
    } else {
      map.removeLayer(wildfiresLayer);
      wildfiresToggleBtn.classList.remove('active');
    }
  });

  const fireIcon = L.divIcon({
    html: "🔥",
    className: "emoji-icon",
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });

  function inNorthAmerica(lat, lng) {
    return lat >= 14 && lat <= 72 && lng >= -168 && lng <= -52;
  }

  async function fetchWildfiresFromFile() {
    try {
      if (fireCountDiv) fireCountDiv.textContent = 'Caricamento incendi...';

      const res = await fetch('static/data/wildfires_latest.csv');
      if (!res.ok) throw new Error(`Errore caricamento file: ${res.status}`);
      const text = await res.text();

      const lines = text.trim().split('\n').filter(Boolean);
      if (lines.length === 0) {
        wildfiresLayer.clearLayers();
        if (fireCountDiv) fireCountDiv.textContent = 'Nessun dato incendi';
        return 0;
      }

      const rows = lines.map(r => r.split(','));
      const header = rows.shift();

      const latIdx = header.indexOf('latitude');
      const lonIdx = header.indexOf('longitude');
      const frpIdx = header.indexOf('frp');
      const brightnessIdx = header.indexOf('bright_ti4') !== -1 ? header.indexOf('bright_ti4') : header.indexOf('brightness');
      const dateIdx = header.indexOf('acq_date');
      const timeIdx = header.indexOf('acq_time');

      if (latIdx === -1 || lonIdx === -1 || frpIdx === -1) {
        console.error('CSV header missing required columns (latitude, longitude, frp)');
        if (fireCountDiv) fireCountDiv.textContent = 'Formato CSV non valido';
        return 0;
      }

      let fires = [];

      for (const r of rows) {
        if (r.length <= Math.max(latIdx, lonIdx, frpIdx)) continue;

        const lat = parseFloat(r[latIdx]);
        const lon = parseFloat(r[lonIdx]);
        const frp = parseFloat(r[frpIdx]);
        const brightness = (brightnessIdx !== -1 && r.length > brightnessIdx) ? parseFloat(r[brightnessIdx]) : NaN;
        const acq_date = (dateIdx !== -1 && r.length > dateIdx) ? r[dateIdx] : '';
        const acq_time = (timeIdx !== -1 && r.length > timeIdx) ? r[timeIdx] : '';

        if (!isNaN(lat) && !isNaN(lon) && !isNaN(frp) && inNorthAmerica(lat, lon)) {
          fires.push({ lat, lon, frp, brightness, acq_date, acq_time });
        }
      }

      fires = fires.sort((a, b) => b.frp - a.frp).slice(0, 40);

      wildfiresLayer.clearLayers();

      fires.forEach(fire => {
        L.marker([fire.lat, fire.lon], { icon: fireIcon }).bindPopup(`
          <b>🔥 Incendio attivo</b><br>
          Brightness: ${Number.isFinite(fire.brightness) ? fire.brightness : 'n/a'}<br>
          FRP: ${fire.frp}<br>
          Data: ${fire.acq_date} ${fire.acq_time}
        `).addTo(wildfiresLayer);
      });

      if (fireCountDiv) fireCountDiv.textContent = `🔥 Fuochi attivi: ${fires.length}`;

      if (wildfiresLayer.getLayers().length > 0 && map.hasLayer(wildfiresLayer)) {
        try { map.fitBounds(wildfiresLayer.getBounds()); }
        catch (e) { console.warn('fitBounds failed for wildfires layer:', e); }
      }

      console.log('Wildfires parsed:', fires.length);
      return fires.length;

    } catch (e) {
      console.error('Errore caricamento incendi:', e);
      if (fireCountDiv) fireCountDiv.textContent = 'Errore caricamento incendi';
      wildfiresLayer.clearLayers();
      return 0;
    }
  }

  // ---------------- AQI + other map stuff ----------------
  const layerGroup = L.layerGroup().addTo(map);
  let currentAbort = null;

  let minAQIFilter = 0;
  const aqiSlider = document.getElementById('aqi-slider');
  if (aqiSlider) {
    aqiSlider.addEventListener('input', () => {
      minAQIFilter = parseInt(aqiSlider.value, 10) || 0;
      filterAQIData();
    });
  }

  let lastAQIData = null;

  function getAQIColor(aqi) {
    if (aqi == null) return '#999999';
    if (aqi <= 50) return '#00e400';
    if (aqi <= 100) return '#ffff00';
    if (aqi <= 150) return '#ff7e00';
    if (aqi <= 200) return '#ff0000';
    if (aqi <= 300) return '#99004c';
    return '#7e0023';
  }
  function fetchAQIForView() {
    if (currentAbort) currentAbort.abort();
    const controller = new AbortController();
    currentAbort = controller;

    const bounds = map.getBounds();
    const bboxParam = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()].join(',');

    const baseurl = window.location.origin; 
    const url = baseurl + `/data/aqi_na_grid.geojson?bbox=${bboxParam}&min_aqi=0`;

    fetch(url, { signal: controller.signal })
      .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then(data => renderData(data))
      .catch(err => { if (err.name !== 'AbortError') console.error('Errore fetch AQI:', err); })
      .finally(() => { currentAbort = null; });
  }

  function filterAQIData() {
    layerGroup.clearLayers();
    if (!lastAQIData || !lastAQIData.features) return;
    lastAQIData.features.forEach(f => {
      const aqi = f.properties.AQI;
      if (aqi == null || isNaN(aqi) || aqi < minAQIFilter) return;

      const coords = f.geometry && f.geometry.coordinates && f.geometry.coordinates[0];
      if (!coords || coords.length < 4) return;
      const centerLat = (coords[0][1] + coords[2][1]) / 2;
      const centerLon = (coords[0][0] + coords[2][0]) / 2;

      const circle = L.circle([centerLat, centerLon], {
        radius: 25000,
        fillColor: getAQIColor(aqi),
        color: '#000',
        weight: 0,
        fillOpacity: 0.6
      }).addTo(layerGroup);

      circle.bindPopup(`AQI: ${aqi.toFixed(1)}`);
    });
  }

  function renderData(data) {
    lastAQIData = data;
    filterAQIData();
    if (!topZonesLoaded) { updateTopZones(data); topZonesLoaded = true; }
  }

  const legendToggle = document.getElementById('legend-toggle');
  const legendContent = document.getElementById('legend-content');
  safeAddListener(legendToggle, 'click', (e) => {
    e.preventDefault(); e.stopPropagation();
    if (legendContent) legendContent.hidden = !legendContent.hidden;
  });

  function reverseGeocode(lat, lon) {
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`;
    return fetch(url)
      .then(res => { if (!res.ok) throw new Error('Geocode error'); return res.json(); })
      .then(data => data.address && (data.address.city || data.address.town || data.address.village || data.address.state || data.address.country) || 'Unknown location')
      .catch(() => 'Unknown location');
  }

  const topZonesList = document.getElementById('top-zones-list');
  let topZonesLoaded = false;

  const schoolsHospitalsLayer = L.layerGroup().addTo(map);
  const schoolIcon = L.divIcon({ html: "🏫", className: "emoji-icon", iconSize: [24,24], iconAnchor: [12,12] });
  const hospitalIcon = L.divIcon({ html: "🏥", className: "emoji-icon", iconSize: [24,24], iconAnchor: [12,12] });

  function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180) * Math.cos(lat2*Math.PI/180) * Math.sin(dLon/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  }

  const geoCache = new Map();

  async function reverseGeocodeCached(lat, lon) {
    const key = `${lat.toFixed(3)},${lon.toFixed(3)}`;
    if (geoCache.has(key)) return geoCache.get(key);
    const city = await reverseGeocode(lat, lon);
    geoCache.set(key, city);
    return city;
  }

  const schoolsHospitalsToggleBtn = document.getElementById('schools-hospitals-toggle');

  safeAddListener(schoolsHospitalsToggleBtn, 'click', () => {
    if (map.hasLayer(schoolsHospitalsLayer)) {
      map.removeLayer(schoolsHospitalsLayer);
      schoolsHospitalsToggleBtn.classList.remove('active');
    } else {
      map.addLayer(schoolsHospitalsLayer);
      schoolsHospitalsToggleBtn.classList.add('active');
    }
  });


  let schoolsData = null;
  let hospitalsData = null;

  async function loadAmenitiesData() {
    try { schoolsData = await fetch('static/data/schools.geojson').then(r => r.json()); }
    catch (err) { console.error('Failed to load schools data:', err); schoolsData = { features: [] }; }

    try { hospitalsData = await fetch('static/data/hospitals.geojson').then(r => r.json()); }
    catch (err) { console.error('Failed to load hospitals data:', err); hospitalsData = { features: [] }; }
  }

  async function updateTopZones(data) {
    if (!data || !data.features) {
      if (topZonesList) topZonesList.innerHTML = '<li>No data</li>';
      return;
    }

    const sorted = data.features
      .filter(f => f.properties && f.properties.AQI != null && !isNaN(f.properties.AQI))
      .sort((a, b) => b.properties.AQI - a.properties.AQI)
      .slice(0, 10);

    const seenCities = new Set();
    const uniqueZones = [];

    for (const f of sorted) {
      if (uniqueZones.length >= 3) break;

      const coords = f.geometry && f.geometry.coordinates && f.geometry.coordinates[0];
      if (!coords || coords.length < 4) continue;
      const centerLat = (coords[0][1] + coords[2][1]) / 2;
      const centerLon = (coords[0][0] + coords[2][0]) / 2;

      const city = await reverseGeocodeCached(centerLat, centerLon);
      if (!seenCities.has(city)) {
        seenCities.add(city);
        uniqueZones.push({ feature: f, city, centerLat, centerLon });
      }
    }

    if (uniqueZones.length === 0) {
      if (topZonesList) topZonesList.innerHTML = '<li>No AQI data available</li>';
      return;
    }

    if (topZonesList) topZonesList.innerHTML = '';
    uniqueZones.forEach(({ feature, city, centerLat, centerLon }) => {
      const aqi = feature.properties.AQI.toFixed(1);
      const li = document.createElement('li');
      li.innerHTML = `<a href="#" class="top-zone-link">${city}: AQI ${aqi}</a>`;
      li.querySelector('a').addEventListener('click', e => {
        e.preventDefault(); e.stopPropagation();
        map.setView([centerLat, centerLon], 10);
      });
      topZonesList && topZonesList.appendChild(li);
    });

    renderAmenitiesNearZones(uniqueZones);
  }

  function renderAmenitiesNearZones(zones) {
    schoolsHospitalsLayer.clearLayers();
    if (!schoolsData || !hospitalsData) return;

    const radiusKm = 30;
    zones.forEach(({ centerLat, centerLon }) => {
      L.circle([centerLat, centerLon], {
        radius: radiusKm * 1000,
        color: 'magenta',
        weight: 1,
        fill: false,
        dashArray: '4 4'
      }).addTo(layerGroup);

      schoolsData.features.forEach(f => {
        const [lon, lat] = f.geometry.coordinates;
        if (haversineDistance(lat, lon, centerLat, centerLon) <= radiusKm) {
          L.marker([lat, lon], { icon: schoolIcon }).bindPopup(`<b>School</b><br>${f.properties.name || 'Unnamed'}`).addTo(schoolsHospitalsLayer);
        }
      });

      hospitalsData.features.forEach(f => {
        const [lon, lat] = f.geometry.coordinates;
        if (haversineDistance(lat, lon, centerLat, centerLon) <= radiusKm) {
          L.marker([lat, lon], { icon: hospitalIcon }).bindPopup(`<b>Hospital</b><br>${f.properties.name || 'Unnamed'}`).addTo(schoolsHospitalsLayer);
        }
      });
    });
  }

  loadAmenitiesData();

  const nasaLogo = document.getElementById('nasa-logo');
  function updateNasaLogo() {
    if (!nasaLogo) return;
    nasaLogo.src = document.body.classList.contains('dark-mode') ? "static/nasa_logo_dark.png" : "static/nasa_logo.png";
  }
  updateNasaLogo();

  
  safeAddListener(darkModeToggle, 'click', () => {
    document.body.classList.toggle('dark-mode');
    updateNasaLogo();
  });

  // ======================================================================
  // 3. Initialization Calls
  // ======================================================================

  // Initialize forecast on page load (uses default DC coords)
  //fetchWeeklyForecast(); 

  map.whenReady(fetchAQIForView);
  map.on('moveend', fetchAQIForView);

  map.whenReady(() => {
    fetchWildfiresFromFile().then(count => console.log('Wildfires loaded:', count));
  });

  try { map.attributionControl.remove(); } catch (e) { }
});