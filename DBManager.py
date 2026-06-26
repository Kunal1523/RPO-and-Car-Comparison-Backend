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
    
    def update_body_type(self, brand_id: str, car_name: str, body_type: str):
        query = """
        UPDATE cars
        SET body_type = %s
        WHERE brand_id = %s
        AND LOWER(name) = LOWER(%s)
        RETURNING id, name, body_type;
        """

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (body_type, brand_id, car_name))
                row = cur.fetchone()
                conn.commit()

        if not row:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "body_type": row[2]
        }
    
    def clear_body_type(self, brand_id: str, car_name: str):
        query = """
            UPDATE cars
            SET body_type = NULL,
                sub_body_type = NULL
            WHERE brand_id = %s
            AND LOWER(name)=LOWER(%s)
            RETURNING id
        """

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (brand_id, car_name))
                r = cur.fetchone()
                conn.commit()

        return r is not None
    
    def update_sub_body_type(self, brand_id: str, car_name: str, sub_body_type: str | None):
        query = """
        UPDATE cars
        SET sub_body_type = %s
        WHERE brand_id = %s
        AND LOWER(name) = LOWER(%s)
        RETURNING id, name, sub_body_type;
        """

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (sub_body_type, brand_id, car_name))
                row = cur.fetchone()
                conn.commit()

        if not row:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "sub_body_type": row[2]
        }


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

    # def get_feature_master_category_wise(self):
    #     query = "SELECT id, name, category FROM features_master WHERE is_active = true ORDER BY category, name;"
    #     with self.get_conn() as conn:
    #         with conn.cursor() as cursor:
    #             cursor.execute(query)
    #             rows = cursor.fetchall()

    #     result = {}
    #     for feature_id, name, category in rows:
    #         if category not in result:
    #             result[category] = []
    #         result[category].append({
    #             "id": str(feature_id),
    #             "name": name
    #         })
    #     return result
    def get_feature_master_category_wise(self):
        query = """
        SELECT id, name, category, mapped_to_id,
               EXISTS (SELECT 1 FROM features_master f2 WHERE f2.mapped_to_id = features_master.id) AS is_merged,
               COALESCE(sort_order, 0) as sort_order
        FROM features_master
        WHERE is_active = true
        AND mapped_to_id IS NULL
        AND name NOT IN (
            'Wheels Type',
            'Lighting LED tail lamps',
            'Dual tone pack (Black painted) Oustide door mirrors',
            'Electronic Stability Program (ESP) with Hill Hold Control',
            'ABS',
            'ABS with EBD and Brake Assist',
            'Reverse Parking Camera',
            'Seat Belt Reminder',
            'EBD'
        )
        ORDER BY category, COALESCE(sort_order, 0), name;
        """

        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

        result = {}
        for feature_id, name, category, mapped_to_id, is_merged, sort_order in rows:
            if category not in result:
                result[category] = []

            result[category].append({
                "id": str(feature_id),
                "name": name,
                "isMerged": is_merged,
                "sort_order": sort_order
            })

        return result

    def get_all_master_features_flat(self):
        query = """
        SELECT id, name, category, COALESCE(sort_order, 0) as sort_order
        FROM features_master
        WHERE is_active = true
        AND mapped_to_id IS NULL
        AND name NOT IN (
            'Wheels Type',
            'Lighting LED tail lamps',
            'Dual tone pack (Black painted) Oustide door mirrors',
            'Electronic Stability Program (ESP) with Hill Hold Control',
            'ABS',
            'ABS with EBD and Brake Assist',
            'Reverse Parking Camera',
            'Seat Belt Reminder',
            'EBD'
        )
        ORDER BY category, COALESCE(sort_order, 0), name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        
        return [{"feature_id": str(r[0]), "feature_name": r[1], "category": r[2]} for r in rows]

    def soft_delete_feature_master(self, feature_id: str):
        query = "UPDATE features_master SET is_active = false WHERE id = %s"
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (feature_id,))
                if cursor.rowcount == 0:
                    raise Exception("Feature not found or already deleted")
            # autocommit=True
        return {"success": True, "message": "Feature deleted successfully"}

    def reorder_features(self, updates: list):
        """Bulk update sort_order for features. updates = [{id, sort_order}]"""
        if not updates:
            return {"success": True, "updated": 0}
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                for item in updates:
                    cursor.execute(
                        "UPDATE features_master SET sort_order = %s WHERE id = %s",
                        (item["sort_order"], item["id"])
                    )
            conn.commit()
        return {"success": True, "updated": len(updates)}

    def add_feature_master(self, name: str, category: str):
        query = """
        INSERT INTO features_master (name, category, is_active)
        VALUES (%s, %s, true)
        ON CONFLICT (name, category) DO UPDATE
        SET is_active = true, mapped_to_id = NULL
        RETURNING id, name, category;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (name, category))
                row = cursor.fetchone()
                if row:
                    return {"id": str(row[0]), "name": row[1], "category": row[2]}
                return None

    def merge_features(self, feature_ids: list, target_name: str, target_category: str):
        if not feature_ids:
            raise ValueError("No feature IDs provided for merging")
            
        # 1. Ensure the target feature exists
        target_feature = self.add_feature_master(target_name, target_category)
        if not target_feature:
            # If it already existed, fetch it
            query = "SELECT id, name, category FROM features_master WHERE name = %s AND category = %s"
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (target_name, target_category))
                    row = cursor.fetchone()
                    if row:
                        target_feature = {"id": str(row[0]), "name": row[1], "category": row[2]}
                    else:
                        raise Exception("Failed to find or create target feature")
                        
        target_id = target_feature["id"]
        
        # 2. Update mapped_to_id for the given feature_ids
        # Exclude the target_id itself just in case
        ids_to_update = [fid for fid in feature_ids if fid != target_id]
        if not ids_to_update:
            return target_feature
            
        format_strings = ','.join(['%s'] * len(ids_to_update))
        update_query = f"UPDATE features_master SET mapped_to_id = %s WHERE id IN ({format_strings})"
        
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, [target_id] + ids_to_update)
            conn.commit()
            
        return target_feature

    def unmerge_features(self, parent_feature_id: str):
        query = "UPDATE features_master SET mapped_to_id = NULL WHERE mapped_to_id = %s"
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (parent_feature_id,))
            conn.commit()
        return {"success": True, "message": "Features unmerged successfully"}

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
            SELECT vf.id, vf.variant_id, COALESCE(target.id, fm.id) AS feature_id, vf.value,
                   vf.original_name, vf.version, COALESCE(target.name, fm.name) AS feature_name, COALESCE(target.category, fm.category) AS category
            FROM variant_features vf
            JOIN features_master fm ON vf.feature_id = fm.id
            LEFT JOIN features_master target ON fm.mapped_to_id = target.id
            WHERE vf.variant_id = %s AND vf.version = %s AND fm.is_active = true
        """
        params = [variant_id, version]

        if categories:
            placeholders = ','.join(['%s'] * len(categories))
            query += f" AND COALESCE(target.category, fm.category) IN ({placeholders})"
            params.extend(categories)

        query += " ORDER BY COALESCE(target.category, fm.category), COALESCE(target.name, fm.name)"

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

    def rename_feature(self, feature_id: str, new_name: str):
        with self.get_conn() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT name, category FROM features_master WHERE id = %s",
                        (feature_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        raise Exception("Feature not found")
                    old_name, category = row

                    if old_name == new_name:
                        return {"status": "success", "message": "No change"}

                    cur.execute(
                        "SELECT id FROM features_master WHERE name = %s AND category = %s AND is_active = true",
                        (new_name, category)
                    )
                    if cur.fetchone():
                        raise Exception("A feature with this name already exists in this category")

                    cur.execute(
                        "UPDATE features_master SET name = %s WHERE id = %s",
                        (new_name, feature_id)
                    )

                    cur.execute(
                        "UPDATE plan_features SET feature_name = %s WHERE feature_id = %s",
                        (new_name, feature_id)
                    )

                conn.commit()
                return {"status": "success", "message": "Feature renamed successfully"}
            except Exception:
                conn.rollback()
                raise

    def move_feature_category(self, feature_id: str, new_category: str):
        with self.get_conn() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT name, category FROM features_master WHERE id = %s",
                        (feature_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        raise Exception("Feature not found")
                    name, old_category = row

                    if old_category == new_category:
                        return {"status": "success", "message": "No change"}

                    cur.execute(
                        "SELECT id FROM features_master WHERE name = %s AND category = %s AND is_active = true",
                        (name, new_category)
                    )
                    if cur.fetchone():
                        raise Exception("A feature with this name already exists in the target category")

                    cur.execute(
                        "UPDATE features_master SET category = %s WHERE id = %s",
                        (new_category, feature_id)
                    )

                    cur.execute(
                        "UPDATE plan_features SET category = %s WHERE feature_id = %s",
                        (new_category, feature_id)
                    )

                conn.commit()
                return {"status": "success", "message": "Feature moved successfully"}
            except Exception:
                conn.rollback()
                raise

    def deactivate_feature(self, feature_id: str):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE features_master SET is_active = false WHERE id = %s",
                    (feature_id,)
                )
                if cur.rowcount == 0:
                    raise Exception("Feature not found")
        return {"status": "success", "message": "Feature deactivated successfully"}


# ---------------------------------------------------------------------------
# Model Plan
# ---------------------------------------------------------------------------

class ModelPlanDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def get_plan_by_name_and_class(self, name: str, base_variant_class: str):
        query = """
            SELECT id, name, base_variant_class, base_car_id, created_at
            FROM model_plans
            WHERE name = %s AND base_variant_class = %s LIMIT 1;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (name, base_variant_class))
                r = cur.fetchone()
        if not r:
            return None
        return {
            "plan_id": str(r[0]), "name": r[1], "base_variant_class": r[2],
            "base_car_id": str(r[3]), "created_at": r[4].isoformat()
        }

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

class MasterDropdownDbManager(DbManager):
    def get_all(self):
        query = "SELECT id, category, value, is_active FROM master_dropdown_values ORDER BY category, value"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
        result = {}
        for r in rows:
            cat = r[1]
            if cat not in result:
                result[cat] = []
            result[cat].append({"id": str(r[0]), "category": cat, "value": r[2], "is_active": r[3]})
        return result

    def add(self, category: str, value: str):
        query = """
            INSERT INTO master_dropdown_values (category, value)
            VALUES (%s, %s)
            ON CONFLICT (category, value) DO NOTHING
            RETURNING id, category, value, is_active;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (category, value))
                r = cur.fetchone()
                if not r:
                    return None
        return {"id": str(r[0]), "category": r[1], "value": r[2], "is_active": r[3]}

    def delete(self, id: str):
        query = "DELETE FROM master_dropdown_values WHERE id = %s RETURNING id"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (id,))
                r = cur.fetchone()
                if not r:
                    return False
        return True

class NewModelDbManager(DbManager):
    def upsert_nm_variant_feature(self, nm_variant_id: str, feature_id: str, updates: dict):
        """
        UPSERT a feature value/cost_delta for an NM variant.
        If row doesn't exist yet (user typing before paste), it creates it
        by pulling feature_name + category from features_master.
        original_copied_value is NEVER touched here.
        """
        allowed_fields = {"feature_value", "cost_delta"}
        set_clauses = []
        values = []
        for key, val in updates.items():
            if key not in allowed_fields:
                continue
            set_clauses.append(f"{key} = %s")
            values.append(val)
    
        if not set_clauses:
            return None
    
        set_clauses.append("updated_at = now()")
    
        # First try UPDATE
        update_query = f"""
            UPDATE new_model_variant_features
            SET {', '.join(set_clauses)}
            WHERE nm_variant_id = %s AND feature_id = %s
            RETURNING id, feature_id, feature_name, category, feature_value,
                    cost_delta, copied_from_variant_class, original_copied_value;
        """
        update_values = values + [nm_variant_id, feature_id]
    
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(update_query, update_values)
                r = cur.fetchone()
    
                if not r:
                    # Row doesn't exist yet — fetch feature meta from features_master and INSERT
                    cur.execute(
                        "SELECT name, category FROM features_master WHERE id = %s",
                        (feature_id,)
                    )
                    meta = cur.fetchone()
                    if not meta:
                        return None  # feature_id doesn't exist at all → 404
    
                    feature_name, category = meta
    
                    insert_query = """
                        INSERT INTO new_model_variant_features
                            (nm_variant_id, feature_id, feature_name, category,
                            feature_value, cost_delta)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (nm_variant_id, feature_id) DO UPDATE
                            SET feature_value = EXCLUDED.feature_value,
                                cost_delta    = EXCLUDED.cost_delta,
                                updated_at    = now()
                        RETURNING id, feature_id, feature_name, category, feature_value,
                                cost_delta, copied_from_variant_class, original_copied_value;
                    """
                    feature_value = updates.get("feature_value", "")
                    cost_delta    = updates.get("cost_delta", 0)
                    cur.execute(insert_query, (
                        nm_variant_id, feature_id, feature_name, category,
                        feature_value, cost_delta
                    ))
                    r = cur.fetchone()
    
            conn.commit()
    
        if not r:
            return None
    
        return {
            "id": str(r[0]),
            "feature_id": str(r[1]),
            "feature_name": r[2],
            "category": r[3],
            "feature_value": r[4],
            "cost_delta": float(r[5] or 0),
            "copied_from_variant_class": r[6],
            "original_copied_value": r[7],
        }
    def clear_nm_variant_features(self, nm_variant_id: str) -> int:
        """Delete all copied features for an NM variant. Returns count of deleted rows."""
        query = """
            DELETE FROM new_model_variant_features
            WHERE nm_variant_id = %s
            RETURNING id;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (nm_variant_id,))
                deleted = cur.rowcount
            conn.commit()
        return deleted
    def get_all_models(self):
        query_models = "SELECT id, name, body_type, sub_body_type FROM new_models ORDER BY created_at DESC"
        query_variants = "SELECT id, new_model_id, variant_name, engine_type, powertrain_type, drive_type, fuel_type, price FROM new_model_variants ORDER BY created_at"
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query_models)
                models_rows = cur.fetchall()
                cur.execute(query_variants)
                variants_rows = cur.fetchall()

        models = {}
        for r in models_rows:
            models[str(r[0])] = {
                "id": str(r[0]), "name": r[1], "body_type": r[2], "sub_body_type": r[3], "variants": []
            }
        
        for r in variants_rows:
            model_id = str(r[1])
            if model_id in models:
                models[model_id]["variants"].append({
                    "id": str(r[0]),
                    "new_model_id": model_id,
                    "variant_name": r[2],
                    "engine_type": r[3],
                    "powertrain_type": r[4],
                    "drive_type": r[5],
                    "fuel_type": r[6],
                    "price": float(r[7]) if r[7] else None
                })
                
        return list(models.values())

    def create_model(self, name: str, body_type: str, sub_body_type: str):
        query = "INSERT INTO new_models (name, body_type, sub_body_type) VALUES (%s, %s, %s) RETURNING id, name, body_type, sub_body_type"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (name, body_type, sub_body_type))
                r = cur.fetchone()
        return {"id": str(r[0]), "name": r[1], "body_type": r[2], "sub_body_type": r[3], "variants": []}

    def update_model_meta(self, model_id: str, body_type: str, sub_body_type: str):
        query = """
            UPDATE new_models 
            SET body_type = %s, sub_body_type = %s 
            WHERE id = %s 
            RETURNING id, name, body_type, sub_body_type
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (body_type, sub_body_type, model_id))
                r = cur.fetchone()
                if not r:
                    return None
        return {"id": str(r[0]), "name": r[1], "body_type": r[2], "sub_body_type": r[3]}
    def delete_model(self, model_id: str):
        query = """
            DELETE FROM new_models
            WHERE id=%s
            RETURNING id
        """

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (model_id,))
                r = cur.fetchone()

                if not r:
                    return False

        return True
    def add_variant(self, new_model_id: str, variant_name: str, engine_type: str, powertrain_type: str, drive_type: str, fuel_type: str, price: float):
        query = """
            INSERT INTO new_model_variants (new_model_id, variant_name, engine_type, powertrain_type, drive_type, fuel_type, price)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, new_model_id, variant_name, engine_type, powertrain_type, drive_type, fuel_type, price
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (new_model_id, variant_name, engine_type, powertrain_type, drive_type, fuel_type, price))
                r = cur.fetchone()
        return {
            "id": str(r[0]), "new_model_id": str(r[1]), "variant_name": r[2],
            "engine_type": r[3], "powertrain_type": r[4], "drive_type": r[5],
            "fuel_type": r[6], "price": float(r[7]) if r[7] else None
        }

    def update_variant(self, variant_id: str, variant_name: str, engine_type: str, powertrain_type: str, drive_type: str, fuel_type: str, price: float):
        query = """
            UPDATE new_model_variants SET variant_name=%s, engine_type=%s, powertrain_type=%s, drive_type=%s, fuel_type=%s, price=%s, updated_at=now()
            WHERE id=%s
            RETURNING id, new_model_id, variant_name, engine_type, powertrain_type, drive_type, fuel_type, price
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (variant_name, engine_type, powertrain_type, drive_type, fuel_type, price, variant_id))
                r = cur.fetchone()
                if not r:
                    return None
        return {
            "id": str(r[0]), "new_model_id": str(r[1]), "variant_name": r[2],
            "engine_type": r[3], "powertrain_type": r[4], "drive_type": r[5],
            "fuel_type": r[6], "price": float(r[7]) if r[7] else None
        }

    def delete_variant(self, variant_id: str):
        query = "DELETE FROM new_model_variants WHERE id=%s RETURNING id"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (variant_id,))
                r = cur.fetchone()
                if not r:
                    return False
        return True

    # def get_nm_variant_features(self, nm_variant_id: str):
    #     """Get all stored features for a New Model variant."""
    #     query = """
    #         SELECT id, nm_variant_id, feature_id, feature_name, category,
    #                feature_value, sub_variant_values
    #         FROM new_model_variant_features
    #         WHERE nm_variant_id = %s
    #         ORDER BY category, feature_name;
    #     """
    #     with self.get_conn() as conn:
    #         with conn.cursor() as cur:
    #             cur.execute(query, (nm_variant_id,))
    #             rows = cur.fetchall()
    #     return [
    #         {
    #             "id": str(r[0]),
    #             "nm_variant_id": str(r[1]),
    #             "feature_id": str(r[2]),
    #             "feature_name": r[3],
    #             "category": r[4],
    #             "feature_value": r[5] or "",
    #             "sub_variant_values": r[6] or {}
    #         }
    #         for r in rows
    #     ]

    def get_nm_variant_features(self, nm_variant_id: str):
        """Get all stored features for a New Model variant, padded with all active master features."""
        query = """
            SELECT fm.id as master_feature_id, fm.name as master_feature_name, fm.category as master_category,
                   nmf.id, nmf.nm_variant_id, nmf.feature_value, nmf.sub_variant_values, nmf.cost_delta,
                   nmf.copied_from_variant_class, nmf.original_copied_value
            FROM features_master fm
            LEFT JOIN (
                SELECT COALESCE(target.id, src.feature_id) AS mapped_feature_id, src.*
                FROM new_model_variant_features src
                LEFT JOIN features_master fm_src ON src.feature_id = fm_src.id
                LEFT JOIN features_master target ON fm_src.mapped_to_id = target.id
            ) nmf ON fm.id = nmf.mapped_feature_id AND nmf.nm_variant_id = %s
            WHERE fm.is_active = true
              AND fm.mapped_to_id IS NULL
              AND fm.name NOT IN (
                  'Wheels Type', 'Lighting LED tail lamps', 'Dual tone pack (Black painted) Oustide door mirrors',
                  'Electronic Stability Program (ESP) with Hill Hold Control', 'ABS', 'ABS with EBD and Brake Assist',
                  'Reverse Parking Camera', 'Seat Belt Reminder', 'EBD'
              )
            ORDER BY fm.category, COALESCE(fm.sort_order, 0), fm.name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (nm_variant_id,))
                rows = cur.fetchall()

                # Fetch core variant properties that act as features
                cur.execute("""
                    SELECT engine_type, powertrain_type, drive_type, fuel_type
                    FROM new_model_variants
                    WHERE id = %s
                """, (nm_variant_id,))
                core_row = cur.fetchone()

        # Map core properties to standard feature names
        core_overrides = {}
        if core_row:
            core_overrides = {
                "engine type": core_row[0] or "",
                "transmission type": core_row[1] or "",
                "drive type": core_row[2] or "",
                "fuel type": core_row[3] or ""
            }

        result = []
        for r in rows:
            master_feature_id = str(r[0])
            master_feature_name = r[1]
            master_category = r[2]
            
            nmf_id = str(r[3]) if r[3] else None
            feature_value = r[5] or ""

            # Override with core values if applicable
            f_name_lower = master_feature_name.lower().strip()
            if f_name_lower in core_overrides and core_overrides[f_name_lower]:
                feature_value = core_overrides[f_name_lower]
            
            result.append({
                "id": nmf_id,
                "nm_variant_id": nm_variant_id,
                "feature_id": master_feature_id,
                "feature_name": master_feature_name,
                "category": master_category,
                "feature_value": feature_value,
                "sub_variant_values": r[6] or {},
                "cost_delta": float(r[7] or 0),
                "copied_from_variant_class": r[8],
                "original_copied_value": r[9]
            })
        return result

    # def copy_features_from_class(self, nm_variant_id: str, car_id: str, variant_class: str, version: int = 1):
    #     """
    #     Copy features from all sub-variants of a given variant_class (for a car_id)
    #     into new_model_variant_features for nm_variant_id.
    #     For each feature_id, sub_variant_values = {sv_name: value} for all sub-variants.
    #     feature_value is set to the first non-empty value across sub-variants.
    #     """
    #     # 1. Get all sub-variants for the class
    #     sv_query = """
    #         SELECT id, name FROM variants
    #         WHERE car_id = %s AND variant_class = %s AND is_latest = true
    #         ORDER BY name;
    #     """
    #     # 2. For each sub-variant get its features
    #     feat_query = """
    #         SELECT vf.feature_id, vf.value, fm.name AS feature_name, fm.category
    #         FROM variant_features vf
    #         JOIN features_master fm ON vf.feature_id = fm.id
    #         WHERE vf.variant_id = %s AND vf.version = %s AND fm.is_active = true
    #     """

    #     sub_variants = []
    #     features_by_sv = {}  # sv_name -> {feature_id: {value, feature_name, category}}

    #     with self.get_conn() as conn:
    #         with conn.cursor() as cur:
    #             cur.execute(sv_query, (car_id, variant_class))
    #             sv_rows = cur.fetchall()
    #             for sv_id, sv_name in sv_rows:
    #                 sub_variants.append({"id": str(sv_id), "name": sv_name})

    #             for sv in sub_variants:
    #                 cur.execute(feat_query, (sv["id"], version))
    #                 feat_rows = cur.fetchall()
    #                 features_by_sv[sv["name"]] = {
    #                     str(f[0]): {"value": f[1] or "", "feature_name": f[2], "category": f[3]}
    #                     for f in feat_rows
    #                 }

    #     if not sub_variants:
    #         return {"copied": 0, "message": "No sub-variants found for the given class"}

    #     # 3. Collect all unique feature_ids across all sub-variants
    #     all_feature_ids = {}
    #     for sv_name, feats in features_by_sv.items():
    #         for fid, meta in feats.items():
    #             if fid not in all_feature_ids:
    #                 all_feature_ids[fid] = {"feature_name": meta["feature_name"], "category": meta["category"]}

    #     if not all_feature_ids:
    #         return {"copied": 0, "message": "No features found for any sub-variant in this class"}

    #     # 4. Build upsert data
    #     upsert_query = """
    #         INSERT INTO new_model_variant_features
    #             (nm_variant_id, feature_id, feature_name, category, feature_value, sub_variant_values)
    #         VALUES (%s, %s, %s, %s, %s, %s::jsonb)
    #         ON CONFLICT (nm_variant_id, feature_id)
    #         DO UPDATE SET
    #             feature_name = EXCLUDED.feature_name,
    #             category = EXCLUDED.category,
    #             feature_value = EXCLUDED.feature_value,
    #             sub_variant_values = EXCLUDED.sub_variant_values,
    #             updated_at = now();
    #     """

    #     import json
    #     count = 0
    #     with self.get_conn() as conn:
    #         with conn.cursor() as cur:
    #             for fid, meta in all_feature_ids.items():
    #                 # Build per-sub-variant value map
    #                 sv_values = {}
    #                 first_value = ""
    #                 for sv in sub_variants:
    #                     val = features_by_sv[sv["name"]].get(fid, {}).get("value", "")
    #                     sv_values[sv["name"]] = val
    #                     if not first_value and val:
    #                         first_value = val

    #                 cur.execute(upsert_query, (
    #                     nm_variant_id,
    #                     fid,
    #                     meta["feature_name"],
    #                     meta["category"],
    #                     first_value,
    #                     json.dumps(sv_values)
    #                 ))
    #                 count += 1
    #         conn.commit()

    #     return {"copied": count, "sub_variants": [sv["name"] for sv in sub_variants]}


    def copy_features_from_class(self, nm_variant_id: str, car_id: str, variant_class: str, version: int = 1):
        """
        Copy features from all sub-variants of a given variant_class (for a car_id)
        into new_model_variant_features for nm_variant_id.
        For each feature_id, sub_variant_values = {sv_name: value} for all sub-variants.
        feature_value is set to the first non-empty value across sub-variants.
        Tags each row with copied_from_variant_class + original_copied_value (snapshot for edit-detection).
        """
        sv_query = """
            SELECT id, name FROM variants
            WHERE car_id = %s AND variant_class = %s AND is_latest = true
            ORDER BY name;
        """
        feat_query = """
            SELECT vf.feature_id, vf.value, fm.name AS feature_name, fm.category
            FROM variant_features vf
            JOIN features_master fm ON vf.feature_id = fm.id
            WHERE vf.variant_id = %s AND vf.version = %s AND fm.is_active = true
        """

        sub_variants = []
        features_by_sv = {}

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sv_query, (car_id, variant_class))
                sv_rows = cur.fetchall()
                for sv_id, sv_name in sv_rows:
                    sub_variants.append({"id": str(sv_id), "name": sv_name})

                for sv in sub_variants:
                    cur.execute(feat_query, (sv["id"], version))
                    feat_rows = cur.fetchall()
                    features_by_sv[sv["name"]] = {
                        str(f[0]): {"value": f[1] or "", "feature_name": f[2], "category": f[3]}
                        for f in feat_rows
                    }

        if not sub_variants:
            return {"copied": 0, "message": "No sub-variants found for the given class"}

        all_feature_ids = {}
        for sv_name, feats in features_by_sv.items():
            for fid, meta in feats.items():
                if fid not in all_feature_ids:
                    all_feature_ids[fid] = {"feature_name": meta["feature_name"], "category": meta["category"]}

        if not all_feature_ids:
            return {"copied": 0, "message": "No features found for any sub-variant in this class"}

        upsert_query = """
            INSERT INTO new_model_variant_features
                (nm_variant_id, feature_id, feature_name, category, feature_value,
                sub_variant_values, copied_from_variant_class, original_copied_value)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (nm_variant_id, feature_id)
            DO UPDATE SET
                feature_name = EXCLUDED.feature_name,
                category = EXCLUDED.category,
                feature_value = EXCLUDED.feature_value,
                sub_variant_values = EXCLUDED.sub_variant_values,
                copied_from_variant_class = EXCLUDED.copied_from_variant_class,
                original_copied_value = EXCLUDED.original_copied_value,
                updated_at = now();
        """

        import json
        count = 0
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                for fid, meta in all_feature_ids.items():
                    sv_values = {}
                    first_value = ""
                    for sv in sub_variants:
                        val = features_by_sv[sv["name"]].get(fid, {}).get("value", "")
                        sv_values[sv["name"]] = val
                        if not first_value and val:
                            first_value = val

                    cur.execute(upsert_query, (
                        nm_variant_id,
                        fid,
                        meta["feature_name"],
                        meta["category"],
                        first_value,
                        json.dumps(sv_values),
                        variant_class,      # copied_from_variant_class
                        first_value         # original_copied_value (snapshot)
                    ))
                    count += 1
            conn.commit()

        return {"copied": count, "sub_variants": [sv["name"] for sv in sub_variants]}

    # def update_nm_variant_feature(self, nm_variant_id: str, feature_id: str, value: str):
    #     """Update the feature_value for a specific NM variant feature."""
    #     query = """
    #         UPDATE new_model_variant_features
    #         SET feature_value = %s, updated_at = now()
    #         WHERE nm_variant_id = %s AND feature_id = %s
    #         RETURNING id;
    #     """
    #     with self.get_conn() as conn:
    #         with conn.cursor() as cur:
    #             cur.execute(query, (value, nm_variant_id, feature_id))
    #             r = cur.fetchone()
    #         conn.commit()
    #     return bool(r)
    def update_nm_variant_feature(self, nm_variant_id: str, feature_id: str, updates: dict):
        """
        Inline update of feature_value and/or cost_delta for a specific NM variant feature.
        original_copied_value is NEVER modified here — it stays as the copy-time snapshot
        so edits can be detected later (feature_value != original_copied_value).
        """
        allowed_fields = {"feature_value", "cost_delta"}
        set_clauses = []
        values = []
        for key, val in updates.items():
            if key not in allowed_fields:
                continue
            set_clauses.append(f"{key} = %s")
            values.append(val)

        if not set_clauses:
            return None

        set_clauses.append("updated_at = now()")

        query = f"""
            UPDATE new_model_variant_features
            SET {', '.join(set_clauses)}
            WHERE nm_variant_id = %s AND feature_id = %s
            RETURNING id, feature_id, feature_name, category, feature_value,
                    cost_delta, copied_from_variant_class, original_copied_value;
        """
        values.extend([nm_variant_id, feature_id])

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
                r = cur.fetchone()
                conn.commit()

        if not r:
            return None

        return {
            "id": str(r[0]), "feature_id": str(r[1]), "feature_name": r[2], "category": r[3],
            "feature_value": r[4], "cost_delta": float(r[5] or 0),
            "copied_from_variant_class": r[6], "original_copied_value": r[7]
        }

class SidebarDbManager(DbManager):

    def get_sidebar_data(self):
        # These are the exact feature UUIDs from features_master
        FEATURE_IDS = {
            "engine_type":       "2f2c9a92-2933-4d3c-9497-05166c1e3bfd",
            "transmission_type": "e03ef22e-9dd9-497f-a63e-a66498865dec",
            "fuel_type":         "9e7edfff-c83f-4fec-96e9-8dc3a430caa9",
            "drive_type":        "769ad0f5-a7fb-4965-8260-fa1408e11fd7",
        }
    
        # Existing variants — pull price + 4 feature values
        # We use MIN(p.ex_showroom_price) for price (already in paise, divide by 100000)
        # For each of the 4 features, we pick any one distinct value for the variant_class
        # (they tend to be the same across sub-variants; if multiple, we take MIN for text = first alphabetically)
        query_existing = """
            SELECT
                b.name                          AS brand_name,
                c.name                          AS car_name,
                c.body_type,
                v.version,
                v.variant_class,
                MIN(p.ex_showroom_price)        AS ex_showroom_price,
                MAX(CASE WHEN vf.feature_id = %(engine_type)s       THEN vf.value END) AS engine_type,
                MAX(CASE WHEN vf.feature_id = %(transmission_type)s THEN vf.value END) AS transmission_type,
                MAX(CASE WHEN vf.feature_id = %(fuel_type)s         THEN vf.value END) AS fuel_type,
                MAX(CASE WHEN vf.feature_id = %(drive_type)s        THEN vf.value END) AS drive_type
            FROM brands b
            JOIN cars c        ON b.id = c.brand_id
            JOIN variants v    ON c.id = v.car_id
            JOIN pricing p     ON v.id = p.variant_id AND p.is_latest = true
            LEFT JOIN variant_features vf
                ON vf.variant_id = v.id
                AND vf.feature_id IN (
                        %(engine_type)s,
                        %(transmission_type)s,
                        %(fuel_type)s,
                        %(drive_type)s
                    )
                AND vf.version = v.version
            WHERE v.is_latest = true
            GROUP BY b.name, c.name, c.body_type, v.version, v.variant_class
            ORDER BY b.name, c.name, v.variant_class
        """
    
        query_new = """
            SELECT
                'NM'                AS brand_name,
                m.name              AS car_name,
                m.body_type,
                1                   AS version,
                v.variant_name      AS variant_class,
                v.id                AS variant_id,
                v.price             AS ex_showroom_price,
                v.engine_type,
                v.powertrain_type   AS transmission_type,
                v.fuel_type,
                v.drive_type
            FROM new_models m
            JOIN new_model_variants v ON m.id = v.new_model_id
            ORDER BY m.name, v.variant_name
        """
    
        results = []
        with self.get_conn() as conn:
            with conn.cursor() as cur:
    
                # ── Existing models ──────────────────────────────────────────────
                cur.execute(query_existing, FEATURE_IDS)
                for row in cur.fetchall():
                    results.append({
                        "brand":             row[0],
                        "model":             row[1],
                        "body_type":         row[2] if row[2] else "Other",
                        "version":           str(row[3]),
                        "variant":           row[4],
                        "variant_id":        "",
                        "price":             float(row[5]) / 100000.0 if row[5] is not None else 0.0,
                        "engine_type":       row[6] or "",
                        "transmission_type": row[7] or "",
                        "fuel_type":         row[8] or "",
                        "drive_type":        row[9] or "",
                        "is_new_model":      False,
                    })
    
                # ── New Models ───────────────────────────────────────────────────
                cur.execute(query_new)
                for row in cur.fetchall():
                    results.append({
                        "brand":             row[0],
                        "model":             row[1],
                        "body_type":         row[2] if row[2] else "Other",
                        "version":           str(row[3]),
                        "variant":           row[4],
                        "variant_id":        str(row[5]),
                        "price":             float(row[6]) / 100000.0 if row[6] is not None else 0.0,
                        "engine_type":       row[7] or "",
                        "transmission_type": row[8] or "",
                        "fuel_type":         row[9] or "",
                        "drive_type":        row[10] or "",
                        "is_new_model":      True,
                    })
    
        return results


    def get_full_catalog_pricing(self):
        """
        Returns ALL variants (existing + new models) with FULL per-config pricing,
        grouped by variant_class -> list of sub-variant configs.
        No aggregation/collapsing — every fuel/engine/transmission combo is preserved.
        Meant to be loaded ONCE on app load and cached client-side.
        """
        FEATURE_IDS = {
            "engine_type":       "2f2c9a92-2933-4d3c-9497-05166c1e3bfd",
            "transmission_type": "e03ef22e-9dd9-497f-a63e-a66498865dec",
            "fuel_type":         "9e7edfff-c83f-4fec-96e9-8dc3a430caa9",
            "drive_type":        "769ad0f5-a7fb-4965-8260-fa1408e11fd7",
        }

        # ── Existing models: one row per sub-variant (v.id), NOT collapsed by variant_class ──
        query_existing = """
            SELECT
                b.name                          AS brand_name,
                c.name                          AS car_name,
                c.body_type,
                v.version,
                v.variant_class,
                v.id                            AS sub_variant_id,
                p.id                            AS pricing_id,
                p.ex_showroom_price,
                p.currency,
                p.paint_type,
                MAX(CASE WHEN vf.feature_id = %(engine_type)s       THEN vf.value END) AS engine_type,
                MAX(CASE WHEN vf.feature_id = %(transmission_type)s THEN vf.value END) AS transmission_type,
                MAX(CASE WHEN vf.feature_id = %(fuel_type)s         THEN vf.value END) AS fuel_type,
                MAX(CASE WHEN vf.feature_id = %(drive_type)s        THEN vf.value END) AS drive_type
            FROM brands b
            JOIN cars c        ON b.id = c.brand_id
            JOIN variants v    ON c.id = v.car_id
            JOIN pricing p     ON v.id = p.variant_id AND p.is_latest = true
            LEFT JOIN variant_features vf
                ON vf.variant_id = v.id
                AND vf.feature_id IN (
                        %(engine_type)s,
                        %(transmission_type)s,
                        %(fuel_type)s,
                        %(drive_type)s
                    )
                AND vf.version = v.version
            WHERE v.is_latest = true
            GROUP BY b.name, c.name, c.body_type, v.version, v.variant_class, v.id, p.id, p.ex_showroom_price, p.currency, p.paint_type
            ORDER BY b.name, c.name, v.variant_class, p.ex_showroom_price
        """

        query_new = """
            SELECT
                'NM'                AS brand_name,
                m.name              AS car_name,
                m.body_type,
                1                   AS version,
                v.variant_name      AS variant_class,
                v.id                AS sub_variant_id,
                v.id                AS pricing_id,
                v.price             AS ex_showroom_price,
                'INR'               AS currency,
                NULL                AS paint_type,
                v.engine_type,
                v.powertrain_type   AS transmission_type,
                v.fuel_type,
                v.drive_type
            FROM new_models m
            JOIN new_model_variants v ON m.id = v.new_model_id
            ORDER BY m.name, v.variant_name
        """

        # variant_class -> { brand, model, body_type, sub_variants: [...] }
        grouped: dict = {}

        def add_row(row, is_new_model: bool):
            (brand, model, body_type, version, variant_class,
            sub_variant_id, pricing_id, price, currency, paint_type,
            engine_type, transmission_type, fuel_type, drive_type) = row

            key = (brand, model, str(version), variant_class)
            if key not in grouped:
                grouped[key] = {
                    "brand": brand,
                    "model": model,
                    "body_type": body_type or "Other",
                    "version": str(version),
                    "variant_class": variant_class,
                    "is_new_model": is_new_model,
                    "sub_variants": []
                }

            grouped[key]["sub_variants"].append({
                "sub_variant_id": str(sub_variant_id),
                "pricing_id": str(pricing_id),
                "ex_showroom_price": float(price) if price is not None else 0.0,
                "currency": currency or "INR",
                "paint_type": paint_type or "",
                "engine_type": engine_type or "",
                "transmission_type": transmission_type or "",
                "fuel_type": fuel_type or "",
                "drive_type": drive_type or "",
            })

        with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query_existing, FEATURE_IDS)
                    for row in cur.fetchall():
                        add_row(row, is_new_model=False)

                    cur.execute(query_new)
                    for row in cur.fetchall():
                        add_row(row, is_new_model=True)

        return list(grouped.values())
    # def get_sidebar_data(self):
    #     query_existing = """
    #         SELECT 
    #             b.name as brand_name,
    #             c.name as car_name,
    #             c.body_type,
    #             v.version,
    #             v.variant_class,
    #             '' as variant_id,
    #             MIN(p.ex_showroom_price) as ex_showroom_price
    #         FROM brands b
    #         JOIN cars c ON b.id = c.brand_id
    #         JOIN variants v ON c.id = v.car_id
    #         JOIN pricing p ON v.id = p.variant_id
    #         WHERE v.is_latest = true AND p.is_latest = true
    #         GROUP BY b.name, c.name, c.body_type, v.version, v.variant_class
    #     """
        
    #     query_new = """
    #         SELECT
    #             'NM' as brand_name,
    #             m.name as car_name,
    #             m.body_type,
    #             1 as version,
    #             v.variant_name as variant_class,
    #             v.id as variant_id,
    #             v.price as ex_showroom_price
    #         FROM new_models m
    #         JOIN new_model_variants v ON m.id = v.new_model_id
    #     """
        
    #     results = []
    #     with self.get_conn() as conn:
    #         with conn.cursor() as cur:
    #             cur.execute(query_existing)
    #             for row in cur.fetchall():
    #                 results.append({
    #                     "brand": row[0],
    #                     "model": row[1],
    #                     "body_type": row[2] if row[2] else "Other",
    #                     "version": str(row[3]),
    #                     "variant": row[4],
    #                     "variant_id": str(row[5]),
    #                     "price": float(row[6]) / 100000.0 if row[6] is not None else 0.0,
    #                     "is_new_model": False
    #                 })
                    
    #             cur.execute(query_new)
    #             for row in cur.fetchall():
    #                 results.append({
    #                     "brand": row[0],
    #                     "model": row[1],
    #                     "body_type": row[2] if row[2] else "Other",
    #                     "version": str(row[3]),
    #                     "variant": row[4],
    #                     "variant_id": str(row[5]),
    #                     "price": float(row[6]) / 100000.0 if row[6] is not None else 0.0,
    #                     "is_new_model": True
    #                 })
    #     return results