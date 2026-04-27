"""
chatbot/db_schema.py

The canonical schema description injected into LLM prompts.
Instead of passing actual data rows, we give the LLM the schema
so it can write precise, Supabase-compatible SQL queries.
"""

# ──────────────────────────────────────────────────────────────────────────────
# DB Schema passed to LLM for SQL generation
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA = """
-- ============================================================
-- DATABASE: Car Comparison Platform (PostgreSQL / Supabase)
-- ============================================================

-- Brands: top-level automobile manufacturers
CREATE TABLE public.brands (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name       TEXT NOT NULL UNIQUE,          -- e.g. "Hyundai", "Maruti Suzuki"
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Cars: a model under a brand
CREATE TABLE public.cars (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  brand_id    UUID NOT NULL REFERENCES brands(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,                -- e.g. "Creta", "Grand Vitara"
  launch_year INTEGER,
  created_at  TIMESTAMPTZ DEFAULT now(),
  UNIQUE (brand_id, name)
);
CREATE INDEX idx_cars_brand_id ON public.cars (brand_id);

-- Variants: a specific trim/version of a car
-- IMPORTANT: always filter WHERE is_latest = true to get current data
CREATE TABLE public.variants (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  car_id        UUID NOT NULL REFERENCES cars(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,              -- e.g. "S", "SX", "SX (O)"
  version       INTEGER NOT NULL DEFAULT 1,
  is_latest     BOOLEAN NOT NULL DEFAULT true,
  variant_class VARCHAR(50),               -- e.g. "Alpha", "Zeta" (grouping class)
  created_at    TIMESTAMPTZ DEFAULT now(),
  UNIQUE (car_id, name, version)
);
CREATE INDEX idx_variants_car_id ON public.variants (car_id);

-- Pricing: ex-showroom price per variant, optionally split by config
-- IMPORTANT: always filter WHERE is_latest = true to get current prices
-- Price is in INR by default
CREATE TABLE public.pricing (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  variant_id         UUID NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
  ex_showroom_price  NUMERIC NOT NULL,      -- in INR, e.g. 1650000 = ₹16.5 Lakh
  currency           TEXT DEFAULT 'INR',
  version            INTEGER NOT NULL DEFAULT 1,
  is_latest          BOOLEAN NOT NULL DEFAULT true,
  type               TEXT,                  -- e.g. "Standard"
  fuel_type          TEXT,                  -- e.g. "Petrol", "Diesel", "CNG", "Electric"
  engine_type        TEXT,                  -- e.g. "1.5 Turbo", "1.5 NA"
  transmission_type  TEXT,                  -- e.g. "Manual", "Automatic", "DCT", "IVT"
  paint_type         TEXT,                  -- e.g. "standard", "metallic", "dual_tone"
  edition            TEXT,
  created_at         TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_pricing_variant_id ON public.pricing (variant_id);

-- Features master: catalogue of all possible car features
CREATE TABLE public.features_master (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name       TEXT NOT NULL,                -- e.g. "ABS", "Sunroof", "Displacement"
  category   TEXT NOT NULL,               -- e.g. "Safety", "Comfort", "Engine"
  is_active  BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (name, category)
);

-- Variant Features: which features a variant has and their values
-- IMPORTANT: always filter WHERE is_latest = true to get current features
CREATE TABLE public.variant_features (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  variant_id    UUID NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
  feature_id    UUID NOT NULL REFERENCES features_master(id) ON DELETE CASCADE,
  value         TEXT,                      -- e.g. "Yes", "No", "1497 cc", "Standard"
  original_name TEXT,                      -- raw name from source data
  version       INTEGER NOT NULL DEFAULT 1,
  is_latest     BOOLEAN NOT NULL DEFAULT true,
  created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_variant_features_variant_id ON public.variant_features (variant_id);
CREATE INDEX idx_variant_features_feature_id ON public.variant_features (feature_id);

-- ============================================================
-- KEY QUERY PATTERNS (learn these to write correct SQL)
-- ============================================================
-- 1. Get all brands:
--    SELECT id, name FROM brands ORDER BY name;

-- 2. Get cars for a brand:
--    SELECT c.name, c.launch_year FROM cars c JOIN brands b ON b.id=c.brand_id WHERE b.name='Hyundai';

-- 3. Get variants with prices for a car (ALWAYS use is_latest=true):
--    SELECT v.name AS variant, p.ex_showroom_price, p.fuel_type, p.transmission_type
--    FROM variants v JOIN cars c ON c.id=v.car_id JOIN brands b ON b.id=c.brand_id
--    LEFT JOIN pricing p ON p.variant_id=v.id AND p.is_latest=true
--    WHERE b.name='Hyundai' AND c.name='Creta' AND v.is_latest=true
--    ORDER BY p.ex_showroom_price;

-- 4. Compare features of multiple variants:
--    SELECT fm.category, fm.name AS feature, vf.value, v.name AS variant
--    FROM variant_features vf
--    JOIN features_master fm ON fm.id=vf.feature_id
--    JOIN variants v ON v.id=vf.variant_id
--    WHERE vf.variant_id IN ('uuid1','uuid2') AND vf.is_latest=true
--    ORDER BY fm.category, fm.name;

-- 5. Find cheapest variant in a budget:
--    SELECT b.name AS brand, c.name AS car, v.name AS variant,
--           p.ex_showroom_price, p.fuel_type, p.transmission_type
--    FROM pricing p
--    JOIN variants v ON v.id=p.variant_id AND v.is_latest=true
--    JOIN cars c ON c.id=v.car_id JOIN brands b ON b.id=c.brand_id
--    WHERE p.is_latest=true AND p.ex_showroom_price <= 1500000
--    ORDER BY p.ex_showroom_price;

-- ============================================================
-- FEATURE REFERENCE GUIDE (Use these exact names in WHERE fm.name = '...')
-- ============================================================
-- Safety: '360 View Camera', 'ABS with EBD', '6 Airbags', 'Tyre-Pressure Monitoring System (TPMS)', 'Electronic Stability Program (ESP)', 'Reverse Parking Camera', 'Hill Descent Control'
-- Exterior: 'Panaromic Sunroof', 'LED Tail Lamp', 'LED Daytime Running Lamps (DRL)', 'Auto Headlamps', 'Roof Rail', 'Skid Plate (Front & Rear)'
-- Interior: 'Head Up Display', 'Digital Instrument Cluster', 'Ambient Lighting', 'Ventilated Seats', 'Leather Wrapped Steering Wheel'
-- Comfort: 'Automatic Climate Control', 'Cruise Control', 'Wireless Charger', 'Engine Push Start/Stop', 'Rear AC Vents'
-- Connected: 'Suzuki Connect', 'Remote AC Control', 'Live Vehicle Tracking', 'Geofence'
-- Infotainment: 'Wireless Android Auto & Apple CarPlay', 'SmartPlay Pro', 'Surround Sense Powered by ARKAMYS'
-- Dimensions: 'Wheelbase (mm)', 'Ground Clearance', 'Boot Space'
-- ============================================================
"""

# Short version for retry prompts (saves tokens)
SCHEMA_SHORT = """
Tables: brands(id,name) | cars(id,brand_id,name,launch_year) |
variants(id,car_id,name,version,is_latest,variant_class) |
pricing(id,variant_id,ex_showroom_price,currency,is_latest,fuel_type,engine_type,transmission_type,paint_type) |
features_master(id,name,category,is_active) |
variant_features(id,variant_id,feature_id,value,is_latest)

JOIN PATHS:
- variants.car_id -> cars.id -> brands.id
- pricing.variant_id -> variants.id
- variant_features.variant_id -> variants.id
- variant_features.feature_id -> features_master.id

CRITICAL RULES:
- Always filter variants WHERE is_latest=true
- Always filter pricing WHERE is_latest=true
- Always filter variant_features WHERE is_latest=true
- Only SELECT statements allowed
- Use PostgreSQL/Supabase compatible syntax
- Prices are in INR (numeric)
""".strip()
