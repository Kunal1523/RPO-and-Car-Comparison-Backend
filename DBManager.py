# # db_manager.py
# import psycopg2
# from psycopg2 import pool
# from psycopg2 import errors
# import os
# from dotenv import load_dotenv
# import contextlib
# import pdb
# from psycopg2.extras import RealDictCursor
# load_dotenv()



# NORMALIZATION_RULES = {
#     # merge duplicates
#     "Reverse Parking Camera": "Rear Parking Camera",
#     "Seat Belt Reminder-Lamp & Buzzer": "Seat Belt Reminder",
#     "Gear Shift Indicator (Infotainment)": "Gear Shift Indicator",

#     # split composites
#     "ABS with EBD and Brake Assist": [
#         "ABS",
#         "EBD",
#         "Brake Assist"
#     ],
#     "Electronic Stability Program (ESP) with Hill Hold Control": [
#         "Electronic Stability Program (ESP)",
#         "Hill Hold Control"
#     ]
# }

# CATEGORY_REMAP = {
#     "Suzuki Connect": "Connected Car Technology"
# }

# import threading

# # Global pool shared across all DbManager instances
# _pool = None
# _local = threading.local()

# def get_db_pool():
#     global _pool
#     if _pool is None:
#         _pool = pool.ThreadedConnectionPool(
#             1, 10, # min 1, max 10 connections
#             user=os.getenv("user"),
#             password=os.getenv("password"),
#             host=os.getenv("host"),
#             port=os.getenv("port"),
#             dbname=os.getenv("dbname")
#         )
#     return _pool

# class DbManager:
#     def __init__(self):
#         pass

#     def get_conn(self):
#         """
#         Returns a connection for the current thread. 
#         Reuses the same connection within the same thread (e.g. one FastAPI request).
#         """
#         if not hasattr(_local, "conn") or _local.conn is None:
#             _local.conn = get_db_pool().getconn()
#             _local.conn.autocommit = True
#         return _local.conn
    
#     @staticmethod
#     def release_conn():
#         """Releases the connection for the current thread back to the pool."""
#         if hasattr(_local, "conn") and _local.conn is not None:
#             try:
#                 get_db_pool().putconn(_local.conn)
#             except Exception:
#                 pass
#             _local.conn = None

#     @contextlib.contextmanager
#     def connection(self):
#         """Legacy helper for manual context management."""
#         conn = self.get_conn()
#         try:
#             yield conn
#         finally:
#             # We don't release here if we are using thread-local management 
#             # unless we want to be very granular. 
#             # But let's just make it compatible.
#             pass

# class BrandDbManager(DbManager):
#     def __init__(self):
#         super().__init__()

#     def insert_brand(self, brand_name: str):
#         query = """
#         INSERT INTO brands (name)
#         VALUES (%s)
#         ON CONFLICT (name) DO NOTHING
#         RETURNING id, name;
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (brand_name,))
#             result = cursor.fetchone()

#             if result:
#                 return {
#                     "id": result[0],
#                     "name": result[1],
#                     "status": "inserted"
#                 }

#             return {
#                 "name": brand_name,
#                 "status": "already_exists"
#             }

#     def get_brand_id_by_name(self, brand_name: str):
#         query = """
#         SELECT id FROM brands WHERE name = %s;
#         """
#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (brand_name,))
#             result = cursor.fetchone()
#             return result[0] if result else None

#     def get_all_brands(self):
#         query = """
#         SELECT id, name
#         FROM brands
#         ORDER BY name;
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query)
#             rows = cur.fetchall()

#         return [
#             {"id": r[0], "name": r[1]}
#             for r in rows
#         ]
    
# class CarDbManager(DbManager):
#     def __init__(self):
#         super().__init__()

#     def insert_car(self, brand_id: str, car_name: str):
#         query = """
#         INSERT INTO cars (brand_id, name)
#         VALUES (%s, %s)
#         ON CONFLICT (brand_id, name) DO NOTHING
#         RETURNING id, name;
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (brand_id, car_name))
#             result = cursor.fetchone()

#             if result:
#                 return {
#                     "id": result[0],
#                     "name": result[1],
#                     "status": "inserted"
#                 }

#             return {
#                 "name": car_name,
#                 "status": "already_exists"
#             }
    
#     def get_cars_by_brand_id(self, brand_id: str):
#         query = """
#         SELECT id, name
#         FROM cars
#         WHERE brand_id = %s
#         ORDER BY name;
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (brand_id,))
#             rows = cur.fetchall()

#         return [
#             {"id": r[0], "name": r[1]}
#             for r in rows
#         ]

# class VariantDbManager(DbManager):
#     def __init__(self):
#         super().__init__()

#     def get_car_id(self, brand_name: str, car_name: str):
#         query = """
#         SELECT c.id
#         FROM cars c
#         JOIN brands b ON b.id = c.brand_id
#         WHERE b.name = %s AND c.name = %s;
#         """
#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (brand_name, car_name))
#             row = cur.fetchone()
#             return row[0] if row else None

#     def insert_variant(self, car_id: str, variant_name: str, version: int = 1):
#         query = """
#         INSERT INTO variants (car_id, name, version, is_latest)
#         VALUES (%s, %s, %s, true)
#         ON CONFLICT (car_id, name, version) DO NOTHING
#         RETURNING id, name;
#         """
#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (car_id, variant_name, version))
#             row = cur.fetchone()

#             if row:
#                 return {"id": row[0], "name": row[1], "status": "inserted"}

#             return {"name": variant_name, "status": "already_exists"}

#     def bulk_insert_variants(self, car_id: str, variants: list[str]):
#         results = []
#         for v in variants:
#             res = self.insert_variant(car_id, v)
#             results.append(res)
#         return results
    
#     def get_variants_by_brand_and_car(self, brand_name: str, car_name: str):
#         query = """
#         SELECT
#             v.id AS variant_id,
#             v.name AS variant_name,
#             v.version,
#             v.is_latest
#         FROM variants v
#         JOIN cars c ON v.car_id = c.id
#         JOIN brands b ON c.brand_id = b.id
#         WHERE
#             b.name = %s
#             AND c.name = %s
#             AND v.is_latest = true
#         ORDER BY v.name;
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (brand_name, car_name))
#             rows = cursor.fetchall()

#         return [
#             {
#                 "variant_id": row[0],
#                 "variant_name": row[1],
#                 "version": row[2],
#                 "is_latest": row[3]
#             }
#             for row in rows
#         ]
    
#     def get_catalog_data(self):
#         query = """
#         SELECT
#             b.id   AS brand_id,
#             b.name AS brand_name,
#             c.id   AS car_id,
#             c.name AS car_name,
#             v.id   AS variant_id,
#             v.name AS variant_name,
#             v.version
#         FROM brands b
#         JOIN cars c ON c.brand_id = b.id
#         JOIN variants v ON v.car_id = c.id
#         WHERE v.is_latest = true
#         ORDER BY b.name, c.name, v.name;
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query)
#             rows = cursor.fetchall()

#         return [
#             {
#                 "brand_id": row[0],
#                 "brand_name": row[1],
#                 "car_id": row[2],
#                 "car_name": row[3],
#                 "variant_id": row[4],
#                 "variant_name": row[5],
#                 "version": row[6],
#             }
#             for row in rows
#         ]

#     def get_variant_details(self, variant_id: str, version: int = 1):
#         """
#         Get complete variant details including brand and car info
#         """
#         query = """
#             SELECT 
#                 v.id as variant_id,
#                 v.name as variant_name,
#                 v.version,
#                 c.name as car_name,
#                 c.id as car_id,
#                 c.launch_year,
#                 b.name as brand_name,
#                 b.id as brand_id
#             FROM variants v
#             JOIN cars c ON v.car_id = c.id
#             JOIN brands b ON c.brand_id = b.id
#             WHERE v.id = %s AND v.version = %s
#         """
        
#         with self.get_conn().cursor(cursor_factory=RealDictCursor) as cur:
#             cur.execute(query, (variant_id, version))
#             result = cur.fetchone()
#             return dict(result) if result else None

    

#     def get_comparable_variants(
#         self,
#         brand_name=None,
#         car_name=None,
#         price_range_min=None,
#         price_range_max=None,
#         limit=20
#     ):
#         """
#         Get variants that can be compared with optional filters
#         """
#         query = """
#             SELECT 
#                 v.id as variant_id,
#                 v.name as variant_name,
#                 v.version,
#                 c.name as car_name,
#                 c.id as car_id,
#                 b.name as brand_name,
#                 b.id as brand_id,
#                 p.ex_showroom_price,
#                 p.currency,
#                 p.type
#             FROM variants v
#             JOIN cars c ON v.car_id = c.id
#             JOIN brands b ON c.brand_id = b.id
#             LEFT JOIN pricing p ON v.id = p.variant_id AND v.version = p.version
#             WHERE v.is_latest = true
#         """
        
#         params = []
        
#         if brand_name:
#             query += " AND b.name = %s"
#             params.append(brand_name)
        
#         if car_name:
#             query += " AND c.name = %s"
#             params.append(car_name)
        
#         if price_range_min is not None:
#             query += " AND p.ex_showroom_price >= %s"
#             params.append(price_range_min)
        
#         if price_range_max is not None:
#             query += " AND p.ex_showroom_price <= %s"
#             params.append(price_range_max)
        
#         query += " ORDER BY b.name, c.name, p.ex_showroom_price LIMIT %s"
#         params.append(limit)
        
#         with self.get_conn().cursor(cursor_factory=RealDictCursor) as cur:
#             cur.execute(query, tuple(params))
#             results = cur.fetchall()
#             return [dict(row) for row in results]

#     def get_variants_by_car_id(self, car_id: str):
#         query = """
#         SELECT id, name, version, is_latest
#         FROM variants
#         WHERE car_id = %s AND is_latest = true
#         ORDER BY name;
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (car_id,))
#             rows = cur.fetchall()

#         return [
#             {
#                 "id": r[0],
#                 "name": r[1],
#                 "version": r[2],
#                 "is_latest": r[3]
#             }
#             for r in rows
#         ]
    
#     def get_variants_by_class_name_only(self, variant_class: str):
#         """
#         Get all sub-variants under a specific class without needing car_id.
#         car_id is inferred from the class name itself.
#         """
#         query = """
#             SELECT id, name, version, is_latest, car_id
#         FROM variants
#         WHERE variant_class = %s
#         AND is_latest = true
#         ORDER BY name;
#     """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (variant_class,))
#             rows = cur.fetchall()

#         return [
#             {
#                 "id": r[0],
#                 "name": r[1],
#                 "version": r[2],
#                 "is_latest": r[3],
#                 "car_id": r[4]      # ← pulled from DB now
#             }
#             for r in rows
#         ]

#     def get_variant_classes_by_car_id(self, car_id: str):
#             query = """
#             SELECT 
#                 variant_class,
#                 JSON_AGG(
#                     JSON_BUILD_OBJECT(
#                         'id', id,
#                         'name', name,
#                         'version', version,
#                         'is_latest', is_latest
#                     ) ORDER BY name
#                 ) AS variants
#             FROM variants
#             WHERE car_id = %s
#             AND is_latest = true
#             AND variant_class IS NOT NULL
#             GROUP BY variant_class
#             ORDER BY variant_class;
#             """

#             with self.get_conn().cursor() as cur:
#                 cur.execute(query, (car_id,))
#                 rows = cur.fetchall()

#             return [
#                 {
#                     "variant_class": r[0],
#                     "variants": r[1]
#                 }
#                 for r in rows
#             ]
        
#     def get_variants_by_class_name(self, car_id: str, variant_class: str):
#         """
#         Get all sub-variants under a specific class for a car.
#         """
#         query = """
#             SELECT id, name, version, is_latest
#             FROM variants
#             WHERE car_id = %s
#             AND variant_class = %s
#             AND is_latest = true
#             ORDER BY name;
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (car_id, variant_class))
#             rows = cur.fetchall()

#         return [
#             {
#                 "id": r[0],
#                 "name": r[1],
#                 "version": r[2],
#                 "is_latest": r[3]
#             }
#             for r in rows
#         ]

# class PricingDbManager(DbManager):
#     def __init__(self):
#         super().__init__()

#     def bulk_insert_pricing(self, pricing_list: list, version: int = 1):
#         """
#         pricing_list = [
#             {
#                 "variant_id": "...",
#                 "ex_showroom_price": 1972400,
#                 "currency": "INR"
#             },
#             ...
#         ]
#         """

#         if not pricing_list:
#             return {"status": "no_data"}

#         with self.get_conn().cursor() as cursor:
#             for item in pricing_list:
#                 variant_id = item["variant_id"]
#                 price = item["ex_showroom_price"]
#                 currency = item.get("currency", "INR")
#                 type = item.get("type", "Standard")

#                 # Step 1: mark old pricing as not latest
#                 cursor.execute(
#                     """
#                     UPDATE pricing
#                     SET is_latest = false
#                     WHERE variant_id = %s AND is_latest = true;
#                     """,
#                     (variant_id,)
#                 )

#                 # Step 2: insert new pricing
#                 cursor.execute(
#                     """
#                     INSERT INTO pricing (
#                         variant_id,
#                         ex_showroom_price,
#                         currency,
#                         version,
#                         type,
#                         is_latest
#                     )
#                     VALUES (%s, %s, %s, %s, %s,true);
#                     """,
#                     (variant_id, price, currency, version,type)
#                 )

#         return {
#             "status": "success",
#             "records_inserted": len(pricing_list),
#             "version": version
#         }
#     def get_pricing_by_brand_and_car_v1(self, brand_name: str, car_name: str):
#         query = """
#         SELECT
#             v.id AS variant_id,
#             v.name AS variant_name,
#             p.id AS pricing_id,
#             p.ex_showroom_price,
#             p.currency,
#             p.fuel_type,
#             p.engine_type,
#             p.transmission_type,
#             p.paint_type,
#             p.edition,
#             p.version AS pricing_version,
#             p.created_at
#         FROM pricing p
#         JOIN variants v ON p.variant_id = v.id
#         JOIN cars c ON v.car_id = c.id
#         JOIN brands b ON c.brand_id = b.id
#         WHERE
#             b.name = %s
#             AND c.name = %s
#             AND v.is_latest = true
#             AND p.is_latest = true
#         ORDER BY v.name, p.fuel_type, p.transmission_type;
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (brand_name, car_name))
#             rows = cursor.fetchall()

#         return [
#             {
#                 "variant_id": row[0],
#                 "variant_name": row[1],
#                 "pricing_id": row[2],
#                 "ex_showroom_price": float(row[3]) if row[3] else None,
#                 "currency": row[4],
#                 "fuel_type": row[5],
#                 "engine_type": row[6],
#                 "transmission_type": row[7],
#                 "paint_type": row[8],
#                 "edition": row[9],
#                 "pricing_version": row[10],
#                 "created_at": row[11].isoformat() if row[11] else None
#             }
#             for row in rows
#         ]

#     def get_pricing_by_brand_and_car(self, brand_name: str, car_name: str):
#         query = """
#         SELECT
#             v.id AS variant_id,
#             v.name AS variant_name,
#             p.ex_showroom_price,
#             p.currency,
#             p.version AS pricing_version,
#             p.type
#         FROM pricing p
#         JOIN variants v ON p.variant_id = v.id
#         JOIN cars c ON v.car_id = c.id
#         JOIN brands b ON c.brand_id = b.id
#         WHERE
#             b.name = %s
#             AND c.name = %s
#             AND v.is_latest = true
#             AND p.is_latest = true
#         ORDER BY p.ex_showroom_price DESC;
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (brand_name, car_name))
#             rows = cursor.fetchall()

#         return [
#             {
#                 "variant_id": row[0],
#                 "variant_name": row[1],
#                 "ex_showroom_price": float(row[2]) if row[2] else None,
#                 "currency": row[3],
#                 "pricing_version": row[4],
#                 "type": row[5]
#             }
#             for row in rows
#         ]
    
#     def update_existing_price(self, variant_id: int, new_price: float):
#         """Updates the price of the current latest record for a variant."""
#         query = """
#         UPDATE pricing 
#         SET ex_showroom_price = %s 
#         WHERE variant_id = %s AND is_latest = true;
#         """
#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (new_price, variant_id))
#             self.get_conn().commit()
#             return cursor.rowcount > 0

#     def insert_new_price(self, variant_id: int, price: float, p_type: str):
#         """Inserts a new price record as the latest version."""
#         # Note: If version is an integer, we fetch the max and increment it
#         query = """
#         INSERT INTO pricing (variant_id, ex_showroom_price, type, is_latest, currency)
#         VALUES (%s, %s, %s, true, 'INR');
#         """
#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (variant_id, price, p_type))
#             self.get_conn().commit()
#             return True
    
#     def get_price(self, variant_id: str, version: int):
#         """
#         Fetch latest pricing for a variant by version + type
#         """

#         query = """
#         SELECT
#             ex_showroom_price,
#             currency,
#             version,
#             type
#         FROM pricing
#         WHERE
#             variant_id = %s
#             AND version = %s
#             AND is_latest = true
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (variant_id, version))
#             row = cursor.fetchone()

#         if not row:
#             return None

#         return {
#             "ex_showroom_price": float(row[0]) if row[0] is not None else None,
#             "currency": row[1],
#             "version": row[2],
#             "type": row[3],
#         }
#     def get_all_prices(self, variant_id: str, version: int):
#         """
#         Fetch ALL pricing types for a variant (for hover tooltip display)
#         Returns list of all price types (metallic, dual_tone, etc.)
#         """
#         query = """
#         SELECT
#             ex_showroom_price,
#             currency,
#             version,
#             fuel_type,
#             engine_type,
#             transmission_type,
#             paint_type,
#             edition
#         FROM pricing
#         WHERE
#             variant_id = %s
#             AND version = %s
#             AND is_latest = true
#         """

#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query, (variant_id, version))
#             rows = cursor.fetchall()

#         if not rows:
#             return []

#         prices = []
#         for row in rows:
#             prices.append({
#                 "ex_showroom_price": float(row[0]) if row[0] is not None else None,
#                 "currency": row[1],
#                 "version": row[2],
#                 "fuel_type":row[3],
#                 "engine_type":row[4],
#                 "transmission_type":row[5],
#                 "paint_type":row[6],
#                 "edition":row[7]
#             })
        
#         return prices
    
#     def get_price(self, variant_id: str, version: int = 1):
#         """
#         Get pricing for a specific variant and version
#         """
#         query = """
#             SELECT 
#                 variant_id,
#                 ex_showroom_price,
#                 currency,
#                 type,
#                 version
#             FROM pricing
#             WHERE variant_id = %s AND version = %s
#         """
        
#         with self.get_conn().cursor(cursor_factory=RealDictCursor) as cur:
#             cur.execute(query, (variant_id, version))
#             result = cur.fetchone()
#             return dict(result) if result else None

# class FeatureDbManager(DbManager):
#     def __init__(self):
#         super().__init__()

#     # def bulk_insert_features(self, features: list[dict]):
#     #     """
#     #     features = [
#     #         {"name": "Displacement", "category": "Engine"},
#     #         {"name": "Max Power", "category": "Engine"},
#     #         ...
#     #     ]
#     #     """
#     #     pdb.set_trace()
#     #     if not features:
#     #         return {"inserted_count": 0}

#     #     query = """
#     #     INSERT INTO features_master (name, category)
#     #     VALUES (%s, %s)
#     #     ON CONFLICT (name, category) DO NOTHING;
#     #     """

#     #     with self.get_conn().cursor() as cursor:
#     #         for feature in features:
#     #             cursor.execute(
#     #                 query,
#     #                 (feature["name"], feature["category"])
#     #             )

#     #     return {
#     #         "status": "success",
#     #         "inserted_count": len(features)
#     #     }

#     def bulk_insert_features(self, features: list[dict]):
#         """
#         Insert features into features_master.
#         Duplicate (name, category) will be skipped.
#         """

#         if not features:
#             return {
#                 "status": "success",
#                 "inserted_count": 0
#             }

#         query = """
#         INSERT INTO features_master (name, category)
#         VALUES (%s, %s)
#         ON CONFLICT (name, category) DO NOTHING
#         RETURNING id;
#         """

#         inserted_count = 0

#         with self.get_conn().cursor() as cursor:
#             for feature in features:
#                 cursor.execute(
#                     query,
#                     (feature["name"], feature["category"])
#                 )
#                 if cursor.fetchone():
#                     inserted_count += 1

#         self.get_conn().commit()

#         return {
#             "status": "success",
#             "inserted_count": inserted_count
#         }

#     def get_feature_master_category_wise(self):
#         query = """
#             SELECT name, category
#             FROM features_master
#             ORDER BY category, name;
#         """
#         with self.get_conn().cursor() as cursor:
#             cursor.execute(query)
#             rows = cursor.fetchall()

#         result = {}
#         for  name, category in rows:
#             if category not in result:
#                 result[category] = []
#             result[category].append({
#                 "name": name
#             })

#         return result
    
#     def normalize_feature_master(self):
#         conn = self.get_conn()
#         cur = conn.cursor()

#         # 1️⃣ CATEGORY NORMALIZATION
#         for old_cat, new_cat in CATEGORY_REMAP.items():
#             cur.execute("""
#                 UPDATE features_master
#                 SET category = %s
#                 WHERE category = %s
#             """, (new_cat, old_cat))

#         # 2️⃣ SPLIT COMPOSITE FEATURES
#         for composite, split_features in NORMALIZATION_RULES.items():
#             if not isinstance(split_features, list):
#                 continue

#             # deactivate composite
#             cur.execute("""
#                 UPDATE features_master
#                 SET is_active = false
#                 WHERE name = %s
#             """, (composite,))

#             # insert atomic features
#             for feat in split_features:
#                 cur.execute("""
#                     INSERT INTO features_master (name, category)
#                     SELECT %s, category
#                     FROM features_master
#                     WHERE name = %s
#                     LIMIT 1
#                     ON CONFLICT (name, category) DO NOTHING
#                 """, (feat, composite))

#         # 3️⃣ MERGE DUPLICATES
#         for old_name, canonical in NORMALIZATION_RULES.items():
#             if isinstance(canonical, list):
#                 continue

#             # deactivate old
#             cur.execute("""
#                 UPDATE features_master
#                 SET is_active = false
#                 WHERE name = %s
#             """, (old_name,))

#             # ensure canonical exists
#             cur.execute("""
#                 INSERT INTO features_master (name, category)
#                 SELECT %s, category
#                 FROM features_master
#                 WHERE name = %s
#                 LIMIT 1
#                 ON CONFLICT (name, category) DO NOTHING
#             """, (canonical, old_name))

#         conn.commit()
#         return {"status": "feature master normalized"}
   
    
#     def get_variant_features(
#     self,
#     variant_id: str,
#     version: int = 1,
#     categories=None
# ):
#         """
#         Get all features for a variant, optionally filtered by categories
#         """
#         query = """
#             SELECT 
#                 vf.id,
#                 vf.variant_id,
#                 vf.feature_id,
#                 vf.value,
#                 vf.original_name,
#                 vf.version,
#                 fm.name as feature_name,
#                 fm.category
#             FROM variant_features vf
#             JOIN features_master fm ON vf.feature_id = fm.id
#             WHERE vf.variant_id = %s 
#             AND vf.version = %s
#             AND fm.is_active = true
#         """
        
#         params = [variant_id, version]
        
#         if categories:
#             placeholders = ','.join(['%s'] * len(categories))
#             query += f" AND fm.category IN ({placeholders})"
#             params.extend(categories)
        
#         query += " ORDER BY fm.category, fm.name"
        
#         with self.get_conn().cursor(cursor_factory=RealDictCursor) as cur:
#             cur.execute(query, tuple(params))
#             results = cur.fetchall()
#             return [dict(row) for row in results]
        
#     def get_features(self, category=None):
#         query = """
#         SELECT id, name, category
#         FROM features_master
#         WHERE is_active = true
#         """
#         params = []

#         if category:
#             query += " AND category = %s"
#             params.append(category)

#         query += " ORDER BY category, name;"

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, tuple(params))
#             rows = cur.fetchall()

#         return [
#             {
#                 "id": r[0],
#                 "name": r[1],
#                 "category": r[2]
#             }
#             for r in rows
#         ]
    
#     def get_variant_feature_latest(self, variant_id: str, feature_id: str):
#         query = """
#         SELECT value, version
#         FROM variant_features
#         WHERE variant_id = %s
#         AND feature_id = %s
#         AND is_latest = true
#         LIMIT 1;
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (variant_id, feature_id))
#             row = cur.fetchone()

#         if not row:
#             return None

#         return {
#             "value": row[0],
#             "version": row[1]
#         }
    

#     def get_all_features_for_variant(self, variant_id: str):
#         query = """
#         SELECT 
#             fm.id,
#             fm.name,
#             fm.category,
#             vf.value
#         FROM features_master fm
#         LEFT JOIN variant_features vf
#             ON vf.feature_id = fm.id
#             AND vf.variant_id = %s
#             AND vf.is_latest = true
#         WHERE fm.is_active = true
#         ORDER BY fm.category, fm.name;
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (variant_id,))
#             rows = cur.fetchall()

#         return [
#             {
#                 "feature_id": r[0],
#                 "feature_name": r[1],
#                 "category": r[2],
#                 "value": r[3]
#             }
#             for r in rows
#         ]

#     def update_variant_feature_value(self, variant_id: str, feature_id: str, value: str):
#         with self.get_conn().cursor() as cur:

#             # 1️⃣ Try update first
#             cur.execute("""
#                 UPDATE variant_features
#                 SET value = %s
#                 WHERE variant_id = %s
#                 AND feature_id = %s
#                 AND is_latest = TRUE;
#             """, (value, variant_id, feature_id))

#             # 2️⃣ If nothing updated → insert new
#             if cur.rowcount == 0:
#                 cur.execute("""
#                     INSERT INTO variant_features (
#                         variant_id, feature_id, value, is_latest
#                     )
#                     VALUES (%s, %s, %s, TRUE);
#                 """, (variant_id, feature_id, value))

#         self.get_conn().commit()

    
#     def create_feature(self, name: str, category: str, is_active: bool = True):
#         with self.get_conn().cursor() as cur:
#             try:
#                 cur.execute("""
#                     INSERT INTO features_master (
#                         name,
#                         category,
#                         is_active
#                     )
#                     VALUES (%s, %s, %s)
#                     RETURNING id, name, category, is_active, created_at;
#                 """, (name, category, is_active))

#                 row = cur.fetchone()
#                 self.get_conn().commit()

#                 return {
#                     "id": row[0],
#                     "name": row[1],
#                     "category": row[2],
#                     "is_active": row[3],
#                     "created_at": row[4]
#                 }

#             except errors.UniqueViolation:
#                 self.get_conn().rollback()
#                 raise Exception("Feature with this name already exists in this category")

#             except Exception as e:
#                 self.get_conn().rollback()
#                 raise e

#     def get_categories(self):
#         query = """
#         SELECT DISTINCT category
#         FROM features_master
#         WHERE is_active = true
#         AND category IS NOT NULL
#         ORDER BY category;
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query)
#             rows = cur.fetchall()

#         return {
#             "categories": [r[0] for r in rows]
#         }

#     def update_variant_feature(self, variant_id, feature_id, value, version=1):

#         query = """
#             UPDATE variant_features
#             SET value = %s
#             WHERE variant_id = %s
#             AND feature_id = %s
#             AND version = %s
#         """

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (
#                 value,
#                 variant_id,
#                 feature_id,
#                 version
#             ))

#             if cur.rowcount == 0:
#                 return False   # No row updated

#             self.get_conn().commit()
#             return True




# class ModelPlanDbManager(DbManager):
#     def __init__(self):
#         super().__init__()

#     def create_plan(self, name: str, base_variant_class: str, base_car_id: str):
#         query = """
#             INSERT INTO model_plans (name, base_variant_class, base_car_id)
#             VALUES (%s, %s, %s)
#             RETURNING id, name, base_variant_class, base_car_id, created_at;
#         """
#         with self.get_conn() as conn:
#             with conn.cursor() as cur:
#                 cur.execute(query, (name, base_variant_class, base_car_id))
#                 r = cur.fetchone()
#                 conn.commit()
#         return {
#             "plan_id": str(r[0]),
#             "name": r[1],
#             "base_variant_class": r[2],
#             "base_car_id": str(r[3]),
#             "created_at": r[4].isoformat()
#         }

#     def get_plan_by_id(self, plan_id: str):
#         query = """
#             SELECT id, name, base_variant_class, base_car_id, created_at, updated_at
#             FROM model_plans
#             WHERE id = %s;
#         """
#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (plan_id,))
#             r = cur.fetchone()
#         if not r:
#             return None
#         return {
#             "plan_id": str(r[0]),
#             "name": r[1],
#             "base_variant_class": r[2],
#             "base_car_id": str(r[3]),
#             "created_at": r[4].isoformat(),
#             "updated_at": r[5].isoformat()
#         }

#     def rename_plan(self, plan_id: str, new_name: str):
#         query = """
#             UPDATE model_plans
#             SET name = %s, updated_at = now()
#             WHERE id = %s
#             RETURNING id, name;
#         """
#         with self.get_conn() as conn:
#             with conn.cursor() as cur:
#                 cur.execute(query, (new_name, plan_id))
#                 r = cur.fetchone()
#                 conn.commit()
#         if not r:
#             return None
#         return {"plan_id": str(r[0]), "name": r[1]}

#     def list_plans(self, base_variant_class: str = None):
#         if base_variant_class:
#             query = """
#                 SELECT id, name, base_variant_class, base_car_id, created_at
#                 FROM model_plans
#                 WHERE base_variant_class = %s
#                 ORDER BY created_at DESC;
#             """
#             params = (base_variant_class,)
#         else:
#             query = """
#                 SELECT id, name, base_variant_class, base_car_id, created_at
#                 FROM model_plans
#                 ORDER BY created_at DESC;
#             """
#             params = ()

#         with self.get_conn().cursor() as cur:
#             cur.execute(query, params)
#             rows = cur.fetchall()

#         return [
#             {
#                 "plan_id": str(r[0]),
#                 "name": r[1],
#                 "base_variant_class": r[2],
#                 "base_car_id": str(r[3]),
#                 "created_at": r[4].isoformat()
#             }
#             for r in rows
#         ]

#     def delete_plan(self, plan_id: str):
#         query = """
#             DELETE FROM model_plans
#             WHERE id = %s
#             RETURNING name;
#         """
#         with self.get_conn() as conn:
#             with conn.cursor() as cur:
#                 cur.execute(query, (plan_id,))
#                 r = cur.fetchone()
#                 conn.commit()
#         return r[0] if r else None


# class PlanFeatureDbManager(DbManager):
#     def __init__(self):
#         super().__init__()

#     def bulk_insert_inherited_features(self, plan_id: str, features: list):
#         """
#         features: list of { feature_id, feature_name, category, value }
#         Called once when plan is created to copy base class features.
#         """
#         query = """
#             INSERT INTO plan_features
#                 (plan_id, feature_id, feature_name, category, value, original_value, is_inherited, cost_delta, price_delta)
#             VALUES (%s, %s, %s, %s, %s, %s, true, 0, 0);
#         """
#         with self.get_conn() as conn:
#             with conn.cursor() as cur:
#                 cur.executemany(query, [
#                     (plan_id, f["feature_id"], f["feature_name"], f["category"], f.get("value", ""), f.get("value", ""))
#                     for f in features
#                 ])
#                 conn.commit()
#         return len(features)

#     def get_features_by_plan(self, plan_id: str, include_deleted: bool = False):
#         if include_deleted:
#             query = """
#                 SELECT id, feature_id, feature_name, category, value, original_value,
#                        is_inherited, is_deleted, cost_delta, price_delta
#                 FROM plan_features
#                 WHERE plan_id = %s
#                 ORDER BY category, display_order, feature_name;
#             """
#         else:
#             query = """
#                 SELECT id, feature_id, feature_name, category, value, original_value,
#                        is_inherited, is_deleted, cost_delta, price_delta
#                 FROM plan_features
#                 WHERE plan_id = %s AND is_deleted = false
#                 ORDER BY category, display_order, feature_name;
#             """
#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (plan_id,))
#             rows = cur.fetchall()

#         return [
#             {
#                 "plan_feature_id": str(r[0]),
#                 "feature_id": str(r[1]) if r[1] else None,
#                 "feature_name": r[2],
#                 "category": r[3],
#                 "value": r[4],
#                 "original_value": r[5],
#                 "is_inherited": r[6],
#                 "is_deleted": r[7],
#                 "cost_delta": float(r[8] or 0),
#                 "price_delta": float(r[9] or 0)
#             }
#             for r in rows
#         ]

#     def add_custom_feature(self, plan_id: str, feature_name: str, category: str,
#                            value: str = None, cost_delta: float = 0, price_delta: float = 0,
#                            after_feature: str = None):
#         with self.get_conn() as conn:
#             with conn.cursor() as cur:
#                 # 1. Determine the new display_order
#                 new_order = 0
#                 if after_feature:
#                     cur.execute("""
#                         SELECT display_order FROM plan_features 
#                         WHERE plan_id = %s AND category = %s AND feature_name = %s
#                         LIMIT 1
#                     """, (plan_id, category, after_feature))
#                     r_order = cur.fetchone()
#                     if r_order:
#                         new_order = r_order[0] + 1
#                         # Shift existing features
#                         cur.execute("""
#                             UPDATE plan_features 
#                             SET display_order = display_order + 1 
#                             WHERE plan_id = %s AND category = %s AND display_order >= %s
#                         """, (plan_id, category, new_order))
#                 else:
#                     # Append to the end
#                     cur.execute("""
#                         SELECT COALESCE(MAX(display_order), 0) FROM plan_features 
#                         WHERE plan_id = %s AND category = %s
#                     """, (plan_id, category))
#                     max_order = cur.fetchone()[0]
#                     new_order = max_order + 1

#                 # 2. Insert the new feature
#                 query = """
#                     INSERT INTO plan_features
#                         (plan_id, feature_id, feature_name, category, value, original_value, is_inherited, cost_delta, price_delta, display_order)
#                     VALUES (%s, NULL, %s, %s, %s, %s, false, %s, %s, %s)
#                     RETURNING id, feature_name, category, value, original_value, cost_delta, price_delta;
#                 """
#                 cur.execute(query, (plan_id, feature_name, category, value, value, cost_delta, price_delta, new_order))
#                 r = cur.fetchone()
                
#                 # 3. Update feature_order table (as requested by user)
#                 cur.execute("""
#                     INSERT INTO feature_order (feature_name, category, order_index)
#                     VALUES (%s, %s, %s)
#                     ON CONFLICT (feature_name, category) 
#                     DO UPDATE SET order_index = EXCLUDED.order_index;
#                 """, (feature_name, category, new_order))
                
#                 conn.commit()

#         return {
#             "plan_feature_id": str(r[0]),
#             "feature_name": r[1],
#             "category": r[2],
#             "value": r[3],
#             "original_value": r[4],
#             "cost_delta": float(r[5] or 0),
#             "price_delta": float(r[6] or 0),
#             "is_inherited": False
#         }

#     def update_feature(self, plan_id: str, plan_feature_id: str,
#                        value: str = None, cost_delta: float = None, price_delta: float = None,
#                        is_deleted: bool = None):
#         fields = []
#         values = []

#         if value is not None:
#             fields.append("value = %s")
#             values.append(value)
#         if cost_delta is not None:
#             fields.append("cost_delta = %s")
#             values.append(cost_delta)
#         if price_delta is not None:
#             fields.append("price_delta = %s")
#             values.append(price_delta)
#         if is_deleted is not None:
#             fields.append("is_deleted = %s")
#             values.append(is_deleted)

#         if not fields:
#             return None

#         fields.append("updated_at = now()")
#         values.extend([plan_id, plan_feature_id])

#         query = f"""
#             UPDATE plan_features
#             SET {', '.join(fields)}
#             WHERE plan_id = %s AND id = %s
#             RETURNING id, feature_name, value, cost_delta, price_delta, is_deleted;
#         """
#         with self.get_conn() as conn:
#             with conn.cursor() as cur:
#                 cur.execute(query, values)
#                 r = cur.fetchone()
#                 conn.commit()

#         if not r:
#             return None
#         return {
#             "plan_feature_id": str(r[0]),
#             "feature_name": r[1],
#             "value": r[2],
#             "cost_delta": float(r[3] or 0),
#             "price_delta": float(r[4] or 0),
#             "is_deleted": r[5]
#         }

# class ChatHistoryDbManager(DbManager):
#     def __init__(self):
#         super().__init__()
#         # Ensure is_starred column exists
#         try:
#             with self.get_conn().cursor() as cursor:
#                 cursor.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS is_starred BOOLEAN DEFAULT FALSE;")
#         except Exception as e:
#             print(f"Warning: Could not add is_starred column: {e}")

#     def get_or_create_session(self, owner_email: str, session_id: int = None):
#         with self.get_conn().cursor(cursor_factory=RealDictCursor) as cursor:
#             if session_id:
#                 cursor.execute(
#                     "SELECT * FROM chat_sessions WHERE id = %s AND owner_email = %s",
#                     (session_id, owner_email)
#                 )
#                 session = cursor.fetchone()
#                 if session:
#                     return dict(session)

#             # Create new session
#             import time
#             now = int(time.time())
#             cursor.execute(
#                 """
#                 INSERT INTO chat_sessions (title, owner_email, created_at, updated_at)
#                 VALUES (%s, %s, %s, %s)
#                 RETURNING id, title, owner_email, created_at, updated_at
#                 """,
#                 ("New Chat", owner_email, now, now)
#             )
#             new_session = cursor.fetchone()
#             return dict(new_session)

#     def append_message(self, session_id: int, role: str, content: str):
#         import time
#         now = int(time.time())
#         with self.get_conn().cursor() as cursor:
#             cursor.execute(
#                 """
#                 INSERT INTO chat_messages (session_id, role, content, created_at)
#                 VALUES (%s, %s, %s, %s)
#                 """,
#                 (session_id, role, content, now)
#             )
#             # Update session timestamp
#             cursor.execute(
#                 "UPDATE chat_sessions SET updated_at = %s WHERE id = %s",
#                 (now, session_id)
#             )

#     def get_session_history(self, session_id: int):
#         with self.get_conn().cursor(cursor_factory=RealDictCursor) as cursor:
#             cursor.execute(
#                 "SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
#                 (session_id,)
#             )
#             return [dict(r) for r in cursor.fetchall()]

#     def list_user_sessions(self, owner_email: str):
#         with self.get_conn().cursor(cursor_factory=RealDictCursor) as cursor:
#             cursor.execute(
#                 "SELECT id, title, is_starred, updated_at, created_at FROM chat_sessions WHERE owner_email = %s ORDER BY updated_at DESC",
#                 (owner_email,)
#             )
#             return [dict(r) for r in cursor.fetchall()]

#     def rename_session(self, session_id: int, owner_email: str, new_title: str):
#         import time
#         now = int(time.time())
#         with self.get_conn().cursor() as cursor:
#             cursor.execute(
#                 "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE id = %s AND owner_email = %s",
#                 (new_title, now, session_id, owner_email)
#             )

#     def toggle_star_session(self, session_id: int, owner_email: str):
#         with self.get_conn().cursor() as cursor:
#             cursor.execute(
#                 "UPDATE chat_sessions SET is_starred = NOT COALESCE(is_starred, FALSE) WHERE id = %s AND owner_email = %s RETURNING is_starred",
#                 (session_id, owner_email)
#             )
#             result = cursor.fetchone()
#             return result[0] if result else False

#     def delete_session(self, session_id: int, owner_email: str):
#         with self.get_conn().cursor() as cursor:
#             # chat_messages will be deleted by ON DELETE CASCADE if set in SQL, 
#             # but let's be explicit if not sure.
#             cursor.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
#             cursor.execute(
#                 "DELETE FROM chat_sessions WHERE id = %s AND owner_email = %s",
#                 (session_id, owner_email)
#             )

#     def soft_delete_feature(self, plan_id: str, plan_feature_id: str):
#         query = """
#             UPDATE plan_features
#             SET is_deleted = true, updated_at = now()
#             WHERE plan_id = %s AND id = %s
#             RETURNING id, feature_name;
#         """
#         with self.get_conn() as conn:
#             with conn.cursor() as cur:
#                 cur.execute(query, (plan_id, plan_feature_id))
#                 r = cur.fetchone()
#                 conn.commit()
#         return {"plan_feature_id": str(r[0]), "feature_name": r[1]} if r else None

#     def get_delta_summary(self, plan_id: str):
#         query = """
#             SELECT feature_name, cost_delta
#             FROM plan_features
#             WHERE plan_id = %s AND is_deleted = false AND cost_delta != 0
#             ORDER BY cost_delta DESC;
#         """
#         with self.get_conn().cursor() as cur:
#             cur.execute(query, (plan_id,))
#             rows = cur.fetchall()

#         breakdown = [
#             {"feature_name": r[0], "cost_delta": float(r[1])}
#             for r in rows
#         ]
#         total_delta = sum(b["cost_delta"] for b in breakdown)

#         return {
#             "total_delta": total_delta,
#             "delta_direction": "increase" if total_delta > 0 else "decrease" if total_delta < 0 else "neutral",
#             "breakdown": breakdown
#         }



# db_manager.py
import psycopg2
from psycopg2 import pool
from psycopg2 import errors
import os
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
load_dotenv()


NORMALIZATION_RULES = {
    # merge duplicates
    "Reverse Parking Camera": "Rear Parking Camera",
    "Seat Belt Reminder-Lamp & Buzzer": "Seat Belt Reminder",
    "Gear Shift Indicator (Infotainment)": "Gear Shift Indicator",

    # split composites
    "ABS with EBD and Brake Assist": [
        "ABS",
        "EBD",
        "Brake Assist"
    ],
    "Electronic Stability Program (ESP) with Hill Hold Control": [
        "Electronic Stability Program (ESP)",
        "Hill Hold Control"
    ]
}

CATEGORY_REMAP = {
    "Suzuki Connect": "Connected Car Technology"
}

import threading

# Global pool shared across all DbManager instances
_pool = None
_pool_lock = threading.Lock()

def get_db_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            # Double-checked locking
            if _pool is None:
                _pool = pool.ThreadedConnectionPool(
                    1, 10,  # min 1, max 10 connections
                    user=os.getenv("user"),
                    password=os.getenv("password"),
                    host=os.getenv("host"),
                    port=os.getenv("port"),
                    dbname=os.getenv("dbname")
                )
    return _pool


class DbManager:
    def __init__(self):
        pass

    def get_conn(self):
        """
        Returns a context manager that checks out ONE connection from the pool
        and returns it when the `with` block exits.

        Usage (always use as a context manager):
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(...)
                # conn.commit() / rollback here if autocommit=False

        autocommit is set to True so explicit commit() calls are not required
        for single-statement operations.  For multi-statement transactions that
        need atomicity, set conn.autocommit = False inside the block, then
        call conn.commit() / conn.rollback() before exiting.
        """
        class _ManagedConn:
            def __init__(self, db_pool):
                self._pool = db_pool
                self._conn = None

            def __enter__(self):
                self._conn = self._pool.getconn()
                self._conn.autocommit = True
                return self._conn

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self._conn:
                    # If the connection is in a broken transaction state, roll it back
                    # before returning to the pool so the next user gets a clean conn.
                    try:
                        if not self._conn.autocommit and self._conn.status != 0:
                            self._conn.rollback()
                    except Exception:
                        pass
                    self._pool.putconn(self._conn)
                    self._conn = None
                return False  # do not suppress exceptions

        return _ManagedConn(get_db_pool())


# ---------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------

class BrandDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def insert_brand(self, brand_name: str):
        query = """
        INSERT INTO brands (name)
        VALUES (%s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id, name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (brand_name,))
                result = cursor.fetchone()

        if result:
            return {"id": result[0], "name": result[1], "status": "inserted"}
        return {"name": brand_name, "status": "already_exists"}

    def get_brand_id_by_name(self, brand_name: str):
        query = "SELECT id FROM brands WHERE name = %s;"
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (brand_name,))
                result = cursor.fetchone()
        return result[0] if result else None

    def get_all_brands(self):
        query = "SELECT id, name FROM brands ORDER BY name;"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
        return [{"id": r[0], "name": r[1]} for r in rows]


# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

class CarDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def insert_car(self, brand_id: str, car_name: str):
        query = """
        INSERT INTO cars (brand_id, name)
        VALUES (%s, %s)
        ON CONFLICT (brand_id, name) DO NOTHING
        RETURNING id, name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (brand_id, car_name))
                result = cursor.fetchone()

        if result:
            return {"id": result[0], "name": result[1], "status": "inserted"}
        return {"name": car_name, "status": "already_exists"}

    def get_cars_by_brand_id(self, brand_id: str):
        query = """
        SELECT id, name FROM cars
        WHERE brand_id = %s ORDER BY name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (brand_id,))
                rows = cur.fetchall()
        return [{"id": r[0], "name": r[1]} for r in rows]


# ---------------------------------------------------------------------------
# Variant
# ---------------------------------------------------------------------------

class VariantDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def get_car_id(self, brand_name: str, car_name: str):
        query = """
        SELECT c.id
        FROM cars c
        JOIN brands b ON b.id = c.brand_id
        WHERE b.name = %s AND c.name = %s;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (brand_name, car_name))
                row = cur.fetchone()
        return row[0] if row else None

    def insert_variant(self, car_id: str, variant_name: str, version: int = 1):
        query = """
        INSERT INTO variants (car_id, name, version, is_latest)
        VALUES (%s, %s, %s, true)
        ON CONFLICT (car_id, name, version) DO NOTHING
        RETURNING id, name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (car_id, variant_name, version))
                row = cur.fetchone()

        if row:
            return {"id": row[0], "name": row[1], "status": "inserted"}
        return {"name": variant_name, "status": "already_exists"}

    def bulk_insert_variants(self, car_id: str, variants: list[str]):
        results = []
        for v in variants:
            res = self.insert_variant(car_id, v)
            results.append(res)
        return results

    def get_variants_by_brand_and_car(self, brand_name: str, car_name: str):
        query = """
        SELECT v.id, v.name, v.version, v.is_latest
        FROM variants v
        JOIN cars c ON v.car_id = c.id
        JOIN brands b ON c.brand_id = b.id
        WHERE b.name = %s AND c.name = %s AND v.is_latest = true
        ORDER BY v.name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (brand_name, car_name))
                rows = cursor.fetchall()
        return [
            {"variant_id": r[0], "variant_name": r[1], "version": r[2], "is_latest": r[3]}
            for r in rows
        ]

    def get_catalog_data(self):
        query = """
        SELECT b.id, b.name, c.id, c.name, v.id, v.name, v.version
        FROM brands b
        JOIN cars c ON c.brand_id = b.id
        JOIN variants v ON v.car_id = c.id
        WHERE v.is_latest = true
        ORDER BY b.name, c.name, v.name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        return [
            {
                "brand_id": r[0], "brand_name": r[1],
                "car_id": r[2], "car_name": r[3],
                "variant_id": r[4], "variant_name": r[5], "version": r[6],
            }
            for r in rows
        ]

    def get_variant_details(self, variant_id: str, version: int = 1):
        query = """
            SELECT v.id, v.name, v.version, c.name, c.id, c.launch_year, b.name, b.id
            FROM variants v
            JOIN cars c ON v.car_id = c.id
            JOIN brands b ON c.brand_id = b.id
            WHERE v.id = %s AND v.version = %s
        """
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (variant_id, version))
                result = cur.fetchone()
        return dict(result) if result else None

    def get_comparable_variants(
        self,
        brand_name=None,
        car_name=None,
        price_range_min=None,
        price_range_max=None,
        limit=20
    ):
        query = """
            SELECT
                v.id, v.name, v.version,
                c.name, c.id,
                b.name, b.id,
                p.ex_showroom_price, p.currency, p.type
            FROM variants v
            JOIN cars c ON v.car_id = c.id
            JOIN brands b ON c.brand_id = b.id
            LEFT JOIN pricing p ON v.id = p.variant_id AND v.version = p.version
            WHERE v.is_latest = true
        """
        params = []

        if brand_name:
            query += " AND b.name = %s"
            params.append(brand_name)
        if car_name:
            query += " AND c.name = %s"
            params.append(car_name)
        if price_range_min is not None:
            query += " AND p.ex_showroom_price >= %s"
            params.append(price_range_min)
        if price_range_max is not None:
            query += " AND p.ex_showroom_price <= %s"
            params.append(price_range_max)

        query += " ORDER BY b.name, c.name, p.ex_showroom_price LIMIT %s"
        params.append(limit)

        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                results = cur.fetchall()
        return [dict(row) for row in results]

    def get_variants_by_car_id(self, car_id: str):
        query = """
        SELECT id, name, version, is_latest
        FROM variants
        WHERE car_id = %s AND is_latest = true
        ORDER BY name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (car_id,))
                rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "version": r[2], "is_latest": r[3]} for r in rows]

    def get_variants_by_class_name_only(self, variant_class: str):
        query = """
            SELECT id, name, version, is_latest, car_id
            FROM variants
            WHERE variant_class = %s AND is_latest = true
            ORDER BY name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (variant_class,))
                rows = cur.fetchall()
        return [
            {"id": r[0], "name": r[1], "version": r[2], "is_latest": r[3], "car_id": r[4]}
            for r in rows
        ]

    def get_variant_classes_by_car_id(self, car_id: str):
        query = """
        SELECT
            variant_class,
            JSON_AGG(
                JSON_BUILD_OBJECT('id', id, 'name', name, 'version', version, 'is_latest', is_latest)
                ORDER BY name
            ) AS variants
        FROM variants
        WHERE car_id = %s AND is_latest = true AND variant_class IS NOT NULL
        GROUP BY variant_class
        ORDER BY variant_class;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (car_id,))
                rows = cur.fetchall()
        return [{"variant_class": r[0], "variants": r[1]} for r in rows]

    def get_variants_by_class_name(self, car_id: str, variant_class: str):
        query = """
            SELECT id, name, version, is_latest
            FROM variants
            WHERE car_id = %s AND variant_class = %s AND is_latest = true
            ORDER BY name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (car_id, variant_class))
                rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "version": r[2], "is_latest": r[3]} for r in rows]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class PricingDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def bulk_insert_pricing(self, pricing_list: list, version: int = 1):
        if not pricing_list:
            return {"status": "no_data"}

        # Use autocommit=False so the whole batch is one atomic transaction
        with self.get_conn() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cursor:
                    for item in pricing_list:
                        variant_id = item["variant_id"]
                        price = item["ex_showroom_price"]
                        currency = item.get("currency", "INR")
                        p_type = item.get("type", "Standard")

                        cursor.execute(
                            "UPDATE pricing SET is_latest = false WHERE variant_id = %s AND is_latest = true;",
                            (variant_id,)
                        )
                        cursor.execute(
                            """
                            INSERT INTO pricing (variant_id, ex_showroom_price, currency, version, type, is_latest)
                            VALUES (%s, %s, %s, %s, %s, true);
                            """,
                            (variant_id, price, currency, version, p_type)
                        )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {"status": "success", "records_inserted": len(pricing_list), "version": version}

    def get_pricing_by_brand_and_car_v1(self, brand_name: str, car_name: str):
        query = """
        SELECT v.id, v.name, p.id, p.ex_showroom_price, p.currency,
               p.fuel_type, p.engine_type, p.transmission_type,
               p.paint_type, p.edition, p.version, p.created_at
        FROM pricing p
        JOIN variants v ON p.variant_id = v.id
        JOIN cars c ON v.car_id = c.id
        JOIN brands b ON c.brand_id = b.id
        WHERE b.name = %s AND c.name = %s AND v.is_latest = true AND p.is_latest = true
        ORDER BY v.name, p.fuel_type, p.transmission_type;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (brand_name, car_name))
                rows = cursor.fetchall()
        return [
            {
                "variant_id": r[0], "variant_name": r[1], "pricing_id": r[2],
                "ex_showroom_price": float(r[3]) if r[3] else None,
                "currency": r[4], "fuel_type": r[5], "engine_type": r[6],
                "transmission_type": r[7], "paint_type": r[8], "edition": r[9],
                "pricing_version": r[10],
                "created_at": r[11].isoformat() if r[11] else None
            }
            for r in rows
        ]

    def get_pricing_by_brand_and_car(self, brand_name: str, car_name: str):
        query = """
        SELECT v.id, v.name, p.ex_showroom_price, p.currency, p.version, p.type
        FROM pricing p
        JOIN variants v ON p.variant_id = v.id
        JOIN cars c ON v.car_id = c.id
        JOIN brands b ON c.brand_id = b.id
        WHERE b.name = %s AND c.name = %s AND v.is_latest = true AND p.is_latest = true
        ORDER BY p.ex_showroom_price DESC;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (brand_name, car_name))
                rows = cursor.fetchall()
        return [
            {
                "variant_id": r[0], "variant_name": r[1],
                "ex_showroom_price": float(r[2]) if r[2] else None,
                "currency": r[3], "pricing_version": r[4], "type": r[5]
            }
            for r in rows
        ]

    def update_existing_price(self, variant_id: int, new_price: float):
        query = "UPDATE pricing SET ex_showroom_price = %s WHERE variant_id = %s AND is_latest = true;"
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (new_price, variant_id))
                return cursor.rowcount > 0

    def insert_new_price(self, variant_id: int, price: float, p_type: str):
        query = """
        INSERT INTO pricing (variant_id, ex_showroom_price, type, is_latest, currency)
        VALUES (%s, %s, %s, true, 'INR');
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (variant_id, price, p_type))
        return True

    def get_price(self, variant_id: str, version: int = 1):
        query = """
            SELECT variant_id, ex_showroom_price, currency, type, version
            FROM pricing
            WHERE variant_id = %s AND version = %s
        """
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (variant_id, version))
                result = cur.fetchone()
        return dict(result) if result else None

    def get_all_prices(self, variant_id: str, version: int):
        query = """
        SELECT ex_showroom_price, currency, version,
               fuel_type, engine_type, transmission_type, paint_type, edition
        FROM pricing
        WHERE variant_id = %s AND version = %s AND is_latest = true
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (variant_id, version))
                rows = cursor.fetchall()

        return [
            {
                "ex_showroom_price": float(r[0]) if r[0] is not None else None,
                "currency": r[1], "version": r[2], "fuel_type": r[3],
                "engine_type": r[4], "transmission_type": r[5],
                "paint_type": r[6], "edition": r[7]
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

class FeatureDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def bulk_insert_features(self, features: list[dict]):
        if not features:
            return {"status": "success", "inserted_count": 0}

        query = """
        INSERT INTO features_master (name, category)
        VALUES (%s, %s)
        ON CONFLICT (name, category) DO NOTHING
        RETURNING id;
        """
        inserted_count = 0
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                for feature in features:
                    cursor.execute(query, (feature["name"], feature["category"]))
                    if cursor.fetchone():
                        inserted_count += 1
            # autocommit=True, no explicit commit needed
        return {"status": "success", "inserted_count": inserted_count}

    def get_feature_master_category_wise(self):
        query = "SELECT name, category FROM features_master ORDER BY category, name;"
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

        result = {}
        for name, category in rows:
            if category not in result:
                result[category] = []
            result[category].append({"name": name})
        return result

    def normalize_feature_master(self):
        with self.get_conn() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    # 1. Category normalization
                    for old_cat, new_cat in CATEGORY_REMAP.items():
                        cur.execute(
                            "UPDATE features_master SET category = %s WHERE category = %s",
                            (new_cat, old_cat)
                        )

                    # 2. Split composite features
                    for composite, split_features in NORMALIZATION_RULES.items():
                        if not isinstance(split_features, list):
                            continue
                        cur.execute(
                            "UPDATE features_master SET is_active = false WHERE name = %s",
                            (composite,)
                        )
                        for feat in split_features:
                            cur.execute("""
                                INSERT INTO features_master (name, category)
                                SELECT %s, category FROM features_master WHERE name = %s LIMIT 1
                                ON CONFLICT (name, category) DO NOTHING
                            """, (feat, composite))

                    # 3. Merge duplicates
                    for old_name, canonical in NORMALIZATION_RULES.items():
                        if isinstance(canonical, list):
                            continue
                        cur.execute(
                            "UPDATE features_master SET is_active = false WHERE name = %s",
                            (old_name,)
                        )
                        cur.execute("""
                            INSERT INTO features_master (name, category)
                            SELECT %s, category FROM features_master WHERE name = %s LIMIT 1
                            ON CONFLICT (name, category) DO NOTHING
                        """, (canonical, old_name))

                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {"status": "feature master normalized"}

    def get_variant_features(self, variant_id: str, version: int = 1, categories=None):
        query = """
            SELECT vf.id, vf.variant_id, vf.feature_id, vf.value,
                   vf.original_name, vf.version, fm.name AS feature_name, fm.category
            FROM variant_features vf
            JOIN features_master fm ON vf.feature_id = fm.id
            WHERE vf.variant_id = %s AND vf.version = %s AND fm.is_active = true
        """
        params = [variant_id, version]

        if categories:
            placeholders = ','.join(['%s'] * len(categories))
            query += f" AND fm.category IN ({placeholders})"
            params.extend(categories)

        query += " ORDER BY fm.category, fm.name"

        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                results = cur.fetchall()
        return [dict(row) for row in results]

    def get_features(self, category=None):
        query = "SELECT id, name, category FROM features_master WHERE is_active = true"
        params = []
        if category:
            query += " AND category = %s"
            params.append(category)
        query += " ORDER BY category, name;"

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "category": r[2]} for r in rows]

    def get_variant_feature_latest(self, variant_id: str, feature_id: str):
        query = """
        SELECT value, version FROM variant_features
        WHERE variant_id = %s AND feature_id = %s AND is_latest = true
        LIMIT 1;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (variant_id, feature_id))
                row = cur.fetchone()
        return {"value": row[0], "version": row[1]} if row else None

    def get_all_features_for_variant(self, variant_id: str):
        query = """
        SELECT fm.id, fm.name, fm.category, vf.value
        FROM features_master fm
        LEFT JOIN variant_features vf
            ON vf.feature_id = fm.id AND vf.variant_id = %s AND vf.is_latest = true
        WHERE fm.is_active = true
        ORDER BY fm.category, fm.name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (variant_id,))
                rows = cur.fetchall()
        return [
            {"feature_id": r[0], "feature_name": r[1], "category": r[2], "value": r[3]}
            for r in rows
        ]

    def update_variant_feature_value(self, variant_id: str, feature_id: str, value: str):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE variant_features SET value = %s
                    WHERE variant_id = %s AND feature_id = %s AND is_latest = TRUE;
                """, (value, variant_id, feature_id))

                if cur.rowcount == 0:
                    cur.execute("""
                        INSERT INTO variant_features (variant_id, feature_id, value, is_latest)
                        VALUES (%s, %s, %s, TRUE);
                    """, (variant_id, feature_id, value))
            # autocommit=True handles the commit

    def create_feature(self, name: str, category: str, is_active: bool = True):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        INSERT INTO features_master (name, category, is_active)
                        VALUES (%s, %s, %s)
                        RETURNING id, name, category, is_active, created_at;
                    """, (name, category, is_active))
                    row = cur.fetchone()
                    # autocommit=True, no explicit commit needed
                    return {
                        "id": row[0], "name": row[1], "category": row[2],
                        "is_active": row[3], "created_at": row[4]
                    }
                except errors.UniqueViolation:
                    raise Exception("Feature with this name already exists in this category")

    def get_categories(self):
        query = """
        SELECT DISTINCT category FROM features_master
        WHERE is_active = true AND category IS NOT NULL
        ORDER BY category;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
        return {"categories": [r[0] for r in rows]}

    def update_variant_feature(self, variant_id, feature_id, value, version=1):
        query = """
            UPDATE variant_features SET value = %s
            WHERE variant_id = %s AND feature_id = %s AND version = %s
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (value, variant_id, feature_id, version))
                return cur.rowcount > 0
            # autocommit=True handles the commit


# ---------------------------------------------------------------------------
# Model Plan
# ---------------------------------------------------------------------------

class ModelPlanDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def create_plan(self, name: str, base_variant_class: str, base_car_id: str):
        query = """
            INSERT INTO model_plans (name, base_variant_class, base_car_id)
            VALUES (%s, %s, %s)
            RETURNING id, name, base_variant_class, base_car_id, created_at;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (name, base_variant_class, base_car_id))
                r = cur.fetchone()
        return {
            "plan_id": str(r[0]), "name": r[1],
            "base_variant_class": r[2], "base_car_id": str(r[3]),
            "created_at": r[4].isoformat()
        }

    def get_plan_by_id(self, plan_id: str):
        query = """
            SELECT id, name, base_variant_class, base_car_id, created_at, updated_at
            FROM model_plans WHERE id = %s;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id,))
                r = cur.fetchone()
        if not r:
            return None
        return {
            "plan_id": str(r[0]), "name": r[1], "base_variant_class": r[2],
            "base_car_id": str(r[3]), "created_at": r[4].isoformat(), "updated_at": r[5].isoformat()
        }

    def rename_plan(self, plan_id: str, new_name: str):
        query = """
            UPDATE model_plans SET name = %s, updated_at = now()
            WHERE id = %s RETURNING id, name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (new_name, plan_id))
                r = cur.fetchone()
        return {"plan_id": str(r[0]), "name": r[1]} if r else None

    def list_plans(self, base_variant_class: str = None):
        if base_variant_class:
            query = """
                SELECT id, name, base_variant_class, base_car_id, created_at
                FROM model_plans WHERE base_variant_class = %s ORDER BY created_at DESC;
            """
            params = (base_variant_class,)
        else:
            query = """
                SELECT id, name, base_variant_class, base_car_id, created_at
                FROM model_plans ORDER BY created_at DESC;
            """
            params = ()

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [
            {
                "plan_id": str(r[0]), "name": r[1], "base_variant_class": r[2],
                "base_car_id": str(r[3]), "created_at": r[4].isoformat()
            }
            for r in rows
        ]

    def delete_plan(self, plan_id: str):
        query = "DELETE FROM model_plans WHERE id = %s RETURNING name;"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id,))
                r = cur.fetchone()
        return r[0] if r else None


# ---------------------------------------------------------------------------
# Plan Feature
# ---------------------------------------------------------------------------

class PlanFeatureDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def bulk_insert_inherited_features(self, plan_id: str, features: list):
        query = """
            INSERT INTO plan_features
                (plan_id, feature_id, feature_name, category, value, original_value, is_inherited, cost_delta, price_delta)
            VALUES (%s, %s, %s, %s, %s, %s, true, 0, 0);
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, [
                    (plan_id, f["feature_id"], f["feature_name"], f["category"],
                     f.get("value", ""), f.get("value", ""))
                    for f in features
                ])
        return len(features)

    def get_features_by_plan(self, plan_id: str, include_deleted: bool = False):
        if include_deleted:
            query = """
                SELECT id, feature_id, feature_name, category, value, original_value,
                       is_inherited, is_deleted, cost_delta, price_delta
                FROM plan_features WHERE plan_id = %s
                ORDER BY category, display_order, feature_name;
            """
        else:
            query = """
                SELECT id, feature_id, feature_name, category, value, original_value,
                       is_inherited, is_deleted, cost_delta, price_delta
                FROM plan_features WHERE plan_id = %s AND is_deleted = false
                ORDER BY category, display_order, feature_name;
            """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id,))
                rows = cur.fetchall()
        return [
            {
                "plan_feature_id": str(r[0]),
                "feature_id": str(r[1]) if r[1] else None,
                "feature_name": r[2], "category": r[3], "value": r[4],
                "original_value": r[5], "is_inherited": r[6], "is_deleted": r[7],
                "cost_delta": float(r[8] or 0), "price_delta": float(r[9] or 0)
            }
            for r in rows
        ]

    def add_custom_feature(self, plan_id: str, feature_name: str, category: str,
                           value: str = None, cost_delta: float = 0, price_delta: float = 0,
                           after_feature: str = None):
        with self.get_conn() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    new_order = 0
                    if after_feature:
                        cur.execute("""
                            SELECT display_order FROM plan_features
                            WHERE plan_id = %s AND category = %s AND feature_name = %s LIMIT 1
                        """, (plan_id, category, after_feature))
                        r_order = cur.fetchone()
                        if r_order:
                            new_order = r_order[0] + 1
                            cur.execute("""
                                UPDATE plan_features SET display_order = display_order + 1
                                WHERE plan_id = %s AND category = %s AND display_order >= %s
                            """, (plan_id, category, new_order))
                    else:
                        cur.execute("""
                            SELECT COALESCE(MAX(display_order), 0) FROM plan_features
                            WHERE plan_id = %s AND category = %s
                        """, (plan_id, category))
                        new_order = cur.fetchone()[0] + 1

                    cur.execute("""
                        INSERT INTO plan_features
                            (plan_id, feature_id, feature_name, category, value, original_value,
                             is_inherited, cost_delta, price_delta, display_order)
                        VALUES (%s, NULL, %s, %s, %s, %s, false, %s, %s, %s)
                        RETURNING id, feature_name, category, value, original_value, cost_delta, price_delta;
                    """, (plan_id, feature_name, category, value, value, cost_delta, price_delta, new_order))
                    r = cur.fetchone()

                    cur.execute("""
                        INSERT INTO feature_order (feature_name, category, order_index)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (feature_name, category) DO UPDATE SET order_index = EXCLUDED.order_index;
                    """, (feature_name, category, new_order))

                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "plan_feature_id": str(r[0]), "feature_name": r[1], "category": r[2],
            "value": r[3], "original_value": r[4],
            "cost_delta": float(r[5] or 0), "price_delta": float(r[6] or 0),
            "is_inherited": False
        }

    def update_feature(self, plan_id: str, plan_feature_id: str,
                       value: str = None, cost_delta: float = None,
                       price_delta: float = None, is_deleted: bool = None):
        fields, values = [], []

        if value is not None:
            fields.append("value = %s"); values.append(value)
        if cost_delta is not None:
            fields.append("cost_delta = %s"); values.append(cost_delta)
        if price_delta is not None:
            fields.append("price_delta = %s"); values.append(price_delta)
        if is_deleted is not None:
            fields.append("is_deleted = %s"); values.append(is_deleted)

        if not fields:
            return None

        fields.append("updated_at = now()")
        values.extend([plan_id, plan_feature_id])

        query = f"""
            UPDATE plan_features SET {', '.join(fields)}
            WHERE plan_id = %s AND id = %s
            RETURNING id, feature_name, value, cost_delta, price_delta, is_deleted;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
                r = cur.fetchone()

        if not r:
            return None
        return {
            "plan_feature_id": str(r[0]), "feature_name": r[1], "value": r[2],
            "cost_delta": float(r[3] or 0), "price_delta": float(r[4] or 0), "is_deleted": r[5]
        }


# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

class ChatHistoryDbManager(DbManager):
    def __init__(self):
        super().__init__()
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS is_starred BOOLEAN DEFAULT FALSE;")
        except Exception as e:
            print(f"Warning: Could not add is_starred column: {e}")

    def get_or_create_session(self, owner_email: str, session_id: int = None):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if session_id:
                    cursor.execute(
                        "SELECT * FROM chat_sessions WHERE id = %s AND owner_email = %s",
                        (session_id, owner_email)
                    )
                    session = cursor.fetchone()
                    if session:
                        return dict(session)

                import time
                now = int(time.time())
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (title, owner_email, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, title, owner_email, created_at, updated_at
                    """,
                    ("New Chat", owner_email, now, now)
                )
                new_session = cursor.fetchone()
                return dict(new_session)

    def append_message(self, session_id: int, role: str, content: str):
        import time
        now = int(time.time())
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
                    (session_id, role, content, now)
                )
                cursor.execute(
                    "UPDATE chat_sessions SET updated_at = %s WHERE id = %s",
                    (now, session_id)
                )

    def get_session_history(self, session_id: int):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT role, content FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
                    (session_id,)
                )
                return [dict(r) for r in cursor.fetchall()]

    def list_user_sessions(self, owner_email: str):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id, title, is_starred, updated_at, created_at FROM chat_sessions WHERE owner_email = %s ORDER BY updated_at DESC",
                    (owner_email,)
                )
                return [dict(r) for r in cursor.fetchall()]

    def rename_session(self, session_id: int, owner_email: str, new_title: str):
        import time
        now = int(time.time())
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE id = %s AND owner_email = %s",
                    (new_title, now, session_id, owner_email)
                )

    def toggle_star_session(self, session_id: int, owner_email: str):
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE chat_sessions SET is_starred = NOT COALESCE(is_starred, FALSE) WHERE id = %s AND owner_email = %s RETURNING is_starred",
                    (session_id, owner_email)
                )
                result = cursor.fetchone()
                return result[0] if result else False

    def delete_session(self, session_id: int, owner_email: str):
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
                cursor.execute(
                    "DELETE FROM chat_sessions WHERE id = %s AND owner_email = %s",
                    (session_id, owner_email)
                )

    def soft_delete_feature(self, plan_id: str, plan_feature_id: str):
        query = """
            UPDATE plan_features SET is_deleted = true, updated_at = now()
            WHERE plan_id = %s AND id = %s RETURNING id, feature_name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id, plan_feature_id))
                r = cur.fetchone()
        return {"plan_feature_id": str(r[0]), "feature_name": r[1]} if r else None

    def get_delta_summary(self, plan_id: str):
        query = """
            SELECT feature_name, cost_delta FROM plan_features
            WHERE plan_id = %s AND is_deleted = false AND cost_delta != 0
            ORDER BY cost_delta DESC;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id,))
                rows = cur.fetchall()

        breakdown = [{"feature_name": r[0], "cost_delta": float(r[1])} for r in rows]
        total_delta = sum(b["cost_delta"] for b in breakdown)
        return {
            "total_delta": total_delta,
            "delta_direction": "increase" if total_delta > 0 else "decrease" if total_delta < 0 else "neutral",
            "breakdown": breakdown
        }