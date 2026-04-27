# db_manager.py
import psycopg2
from psycopg2 import errors
import os
from dotenv import load_dotenv
import pdb
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

class DbManager:
    def __init__(self):
        self.connect()

    def connect(self):
        self.conn = psycopg2.connect(
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port"),
            dbname=os.getenv("dbname")
        )
        self.conn.autocommit = True
    
    def get_conn(self):
        try:
            # Ping connection
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            print("🔄 Reconnecting to database...")
            self.connect()

        return self.conn

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

        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (brand_name,))
            result = cursor.fetchone()

            if result:
                return {
                    "id": result[0],
                    "name": result[1],
                    "status": "inserted"
                }

            return {
                "name": brand_name,
                "status": "already_exists"
            }

    def get_brand_id_by_name(self, brand_name: str):
        query = """
        SELECT id FROM brands WHERE name = %s;
        """
        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (brand_name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_all_brands(self):
        query = """
        SELECT id, name
        FROM brands
        ORDER BY name;
        """

        with self.get_conn().cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

        return [
            {"id": r[0], "name": r[1]}
            for r in rows
        ]
    
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

        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (brand_id, car_name))
            result = cursor.fetchone()

            if result:
                return {
                    "id": result[0],
                    "name": result[1],
                    "status": "inserted"
                }

            return {
                "name": car_name,
                "status": "already_exists"
            }
    
    def get_cars_by_brand_id(self, brand_id: str):
        query = """
        SELECT id, name
        FROM cars
        WHERE brand_id = %s
        ORDER BY name;
        """

        with self.get_conn().cursor() as cur:
            cur.execute(query, (brand_id,))
            rows = cur.fetchall()

        return [
            {"id": r[0], "name": r[1]}
            for r in rows
        ]

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
        with self.get_conn().cursor() as cur:
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
        with self.get_conn().cursor() as cur:
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
        SELECT
            v.id AS variant_id,
            v.name AS variant_name,
            v.version,
            v.is_latest
        FROM variants v
        JOIN cars c ON v.car_id = c.id
        JOIN brands b ON c.brand_id = b.id
        WHERE
            b.name = %s
            AND c.name = %s
            AND v.is_latest = true
        ORDER BY v.name;
        """

        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (brand_name, car_name))
            rows = cursor.fetchall()

        return [
            {
                "variant_id": row[0],
                "variant_name": row[1],
                "version": row[2],
                "is_latest": row[3]
            }
            for row in rows
        ]
    
    def get_catalog_data(self):
        query = """
        SELECT
            b.id   AS brand_id,
            b.name AS brand_name,
            c.id   AS car_id,
            c.name AS car_name,
            v.id   AS variant_id,
            v.name AS variant_name,
            v.version
        FROM brands b
        JOIN cars c ON c.brand_id = b.id
        JOIN variants v ON v.car_id = c.id
        WHERE v.is_latest = true
        ORDER BY b.name, c.name, v.name;
        """

        with self.get_conn().cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        return [
            {
                "brand_id": row[0],
                "brand_name": row[1],
                "car_id": row[2],
                "car_name": row[3],
                "variant_id": row[4],
                "variant_name": row[5],
                "version": row[6],
            }
            for row in rows
        ]

    def get_variant_details(self, variant_id: str, version: int = 1):
        """
        Get complete variant details including brand and car info
        """
        query = """
            SELECT 
                v.id as variant_id,
                v.name as variant_name,
                v.version,
                c.name as car_name,
                c.id as car_id,
                c.launch_year,
                b.name as brand_name,
                b.id as brand_id
            FROM variants v
            JOIN cars c ON v.car_id = c.id
            JOIN brands b ON c.brand_id = b.id
            WHERE v.id = %s AND v.version = %s
        """
        
        with self.get_conn().cursor(cursor_factory=RealDictCursor) as cur:
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
        """
        Get variants that can be compared with optional filters
        """
        query = """
            SELECT 
                v.id as variant_id,
                v.name as variant_name,
                v.version,
                c.name as car_name,
                c.id as car_id,
                b.name as brand_name,
                b.id as brand_id,
                p.ex_showroom_price,
                p.currency,
                p.type
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
        
        with self.get_conn().cursor(cursor_factory=RealDictCursor) as cur:
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

        with self.get_conn().cursor() as cur:
            cur.execute(query, (car_id,))
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "version": r[2],
                "is_latest": r[3]
            }
            for r in rows
        ]
    
    def get_variants_by_class_name_only(self, variant_class: str):
        """
        Get all sub-variants under a specific class without needing car_id.
        car_id is inferred from the class name itself.
        """
        query = """
            SELECT id, name, version, is_latest, car_id
        FROM variants
        WHERE variant_class = %s
        AND is_latest = true
        ORDER BY name;
    """

        with self.get_conn().cursor() as cur:
            cur.execute(query, (variant_class,))
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "version": r[2],
                "is_latest": r[3],
                "car_id": r[4]      # ← pulled from DB now
            }
            for r in rows
        ]

    def get_variant_classes_by_car_id(self, car_id: str):
            query = """
            SELECT 
                variant_class,
                JSON_AGG(
                    JSON_BUILD_OBJECT(
                        'id', id,
                        'name', name,
                        'version', version,
                        'is_latest', is_latest
                    ) ORDER BY name
                ) AS variants
            FROM variants
            WHERE car_id = %s
            AND is_latest = true
            AND variant_class IS NOT NULL
            GROUP BY variant_class
            ORDER BY variant_class;
            """

            with self.get_conn().cursor() as cur:
                cur.execute(query, (car_id,))
                rows = cur.fetchall()

            return [
                {
                    "variant_class": r[0],
                    "variants": r[1]
                }
                for r in rows
            ]
        
    def get_variants_by_class_name(self, car_id: str, variant_class: str):
        """
        Get all sub-variants under a specific class for a car.
        """
        query = """
            SELECT id, name, version, is_latest
            FROM variants
            WHERE car_id = %s
            AND variant_class = %s
            AND is_latest = true
            ORDER BY name;
        """

        with self.get_conn().cursor() as cur:
            cur.execute(query, (car_id, variant_class))
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "version": r[2],
                "is_latest": r[3]
            }
            for r in rows
        ]

class PricingDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def bulk_insert_pricing(self, pricing_list: list, version: int = 1):
        """
        pricing_list = [
            {
                "variant_id": "...",
                "ex_showroom_price": 1972400,
                "currency": "INR"
            },
            ...
        ]
        """

        if not pricing_list:
            return {"status": "no_data"}

        with self.get_conn().cursor() as cursor:
            for item in pricing_list:
                variant_id = item["variant_id"]
                price = item["ex_showroom_price"]
                currency = item.get("currency", "INR")
                type = item.get("type", "Standard")

                # Step 1: mark old pricing as not latest
                cursor.execute(
                    """
                    UPDATE pricing
                    SET is_latest = false
                    WHERE variant_id = %s AND is_latest = true;
                    """,
                    (variant_id,)
                )

                # Step 2: insert new pricing
                cursor.execute(
                    """
                    INSERT INTO pricing (
                        variant_id,
                        ex_showroom_price,
                        currency,
                        version,
                        type,
                        is_latest
                    )
                    VALUES (%s, %s, %s, %s, %s,true);
                    """,
                    (variant_id, price, currency, version,type)
                )

        return {
            "status": "success",
            "records_inserted": len(pricing_list),
            "version": version
        }
    def get_pricing_by_brand_and_car_v1(self, brand_name: str, car_name: str):
        query = """
        SELECT
            v.id AS variant_id,
            v.name AS variant_name,
            p.id AS pricing_id,
            p.ex_showroom_price,
            p.currency,
            p.fuel_type,
            p.engine_type,
            p.transmission_type,
            p.paint_type,
            p.edition,
            p.version AS pricing_version,
            p.created_at
        FROM pricing p
        JOIN variants v ON p.variant_id = v.id
        JOIN cars c ON v.car_id = c.id
        JOIN brands b ON c.brand_id = b.id
        WHERE
            b.name = %s
            AND c.name = %s
            AND v.is_latest = true
            AND p.is_latest = true
        ORDER BY v.name, p.fuel_type, p.transmission_type;
        """

        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (brand_name, car_name))
            rows = cursor.fetchall()

        return [
            {
                "variant_id": row[0],
                "variant_name": row[1],
                "pricing_id": row[2],
                "ex_showroom_price": float(row[3]) if row[3] else None,
                "currency": row[4],
                "fuel_type": row[5],
                "engine_type": row[6],
                "transmission_type": row[7],
                "paint_type": row[8],
                "edition": row[9],
                "pricing_version": row[10],
                "created_at": row[11].isoformat() if row[11] else None
            }
            for row in rows
        ]

    def get_pricing_by_brand_and_car(self, brand_name: str, car_name: str):
        query = """
        SELECT
            v.id AS variant_id,
            v.name AS variant_name,
            p.ex_showroom_price,
            p.currency,
            p.version AS pricing_version,
            p.type
        FROM pricing p
        JOIN variants v ON p.variant_id = v.id
        JOIN cars c ON v.car_id = c.id
        JOIN brands b ON c.brand_id = b.id
        WHERE
            b.name = %s
            AND c.name = %s
            AND v.is_latest = true
            AND p.is_latest = true
        ORDER BY p.ex_showroom_price DESC;
        """

        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (brand_name, car_name))
            rows = cursor.fetchall()

        return [
            {
                "variant_id": row[0],
                "variant_name": row[1],
                "ex_showroom_price": float(row[2]) if row[2] else None,
                "currency": row[3],
                "pricing_version": row[4],
                "type": row[5]
            }
            for row in rows
        ]
    
    def update_existing_price(self, variant_id: int, new_price: float):
        """Updates the price of the current latest record for a variant."""
        query = """
        UPDATE pricing 
        SET ex_showroom_price = %s 
        WHERE variant_id = %s AND is_latest = true;
        """
        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (new_price, variant_id))
            self.conn.commit()
            return cursor.rowcount > 0

    def insert_new_price(self, variant_id: int, price: float, p_type: str):
        """Inserts a new price record as the latest version."""
        # Note: If version is an integer, we fetch the max and increment it
        query = """
        INSERT INTO pricing (variant_id, ex_showroom_price, type, is_latest, currency)
        VALUES (%s, %s, %s, true, 'INR');
        """
        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (variant_id, price, p_type))
            self.conn.commit()
            return True
    
    def get_price(self, variant_id: str, version: int):
        """
        Fetch latest pricing for a variant by version + type
        """

        query = """
        SELECT
            ex_showroom_price,
            currency,
            version,
            type
        FROM pricing
        WHERE
            variant_id = %s
            AND version = %s
            AND is_latest = true
        """

        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (variant_id, version))
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "ex_showroom_price": float(row[0]) if row[0] is not None else None,
            "currency": row[1],
            "version": row[2],
            "type": row[3],
        }
    def get_all_prices(self, variant_id: str, version: int):
        """
        Fetch ALL pricing types for a variant (for hover tooltip display)
        Returns list of all price types (metallic, dual_tone, etc.)
        """
        query = """
        SELECT
            ex_showroom_price,
            currency,
            version,
            fuel_type,
            engine_type,
            transmission_type,
            paint_type,
            edition
        FROM pricing
        WHERE
            variant_id = %s
            AND version = %s
            AND is_latest = true
        """

        with self.get_conn().cursor() as cursor:
            cursor.execute(query, (variant_id, version))
            rows = cursor.fetchall()

        if not rows:
            return []

        prices = []
        for row in rows:
            prices.append({
                "ex_showroom_price": float(row[0]) if row[0] is not None else None,
                "currency": row[1],
                "version": row[2],
                "fuel_type":row[3],
                "engine_type":row[4],
                "transmission_type":row[5],
                "paint_type":row[6],
                "edition":row[7]
            })
        
        return prices
    
    def get_price(self, variant_id: str, version: int = 1):
        """
        Get pricing for a specific variant and version
        """
        query = """
            SELECT 
                variant_id,
                ex_showroom_price,
                currency,
                type,
                version
            FROM pricing
            WHERE variant_id = %s AND version = %s
        """
        
        with self.get_conn().cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (variant_id, version))
            result = cur.fetchone()
            return dict(result) if result else None

class FeatureDbManager(DbManager):
    def __init__(self):
        super().__init__()

    # def bulk_insert_features(self, features: list[dict]):
    #     """
    #     features = [
    #         {"name": "Displacement", "category": "Engine"},
    #         {"name": "Max Power", "category": "Engine"},
    #         ...
    #     ]
    #     """
    #     pdb.set_trace()
    #     if not features:
    #         return {"inserted_count": 0}

    #     query = """
    #     INSERT INTO features_master (name, category)
    #     VALUES (%s, %s)
    #     ON CONFLICT (name, category) DO NOTHING;
    #     """

    #     with self.get_conn().cursor() as cursor:
    #         for feature in features:
    #             cursor.execute(
    #                 query,
    #                 (feature["name"], feature["category"])
    #             )

    #     return {
    #         "status": "success",
    #         "inserted_count": len(features)
    #     }

    def bulk_insert_features(self, features: list[dict]):
        """
        Insert features into features_master.
        Duplicate (name, category) will be skipped.
        """

        if not features:
            return {
                "status": "success",
                "inserted_count": 0
            }

        query = """
        INSERT INTO features_master (name, category)
        VALUES (%s, %s)
        ON CONFLICT (name, category) DO NOTHING
        RETURNING id;
        """

        inserted_count = 0

        with self.get_conn().cursor() as cursor:
            for feature in features:
                cursor.execute(
                    query,
                    (feature["name"], feature["category"])
                )
                if cursor.fetchone():
                    inserted_count += 1

        self.conn.commit()

        return {
            "status": "success",
            "inserted_count": inserted_count
        }

    def get_feature_master_category_wise(self):
        query = """
            SELECT name, category
            FROM features_master
            ORDER BY category, name;
        """
        with self.get_conn().cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        result = {}
        for  name, category in rows:
            if category not in result:
                result[category] = []
            result[category].append({
                "name": name
            })

        return result
    
    def normalize_feature_master(self):
        conn = self.conn
        cur = conn.cursor()

        # 1️⃣ CATEGORY NORMALIZATION
        for old_cat, new_cat in CATEGORY_REMAP.items():
            cur.execute("""
                UPDATE features_master
                SET category = %s
                WHERE category = %s
            """, (new_cat, old_cat))

        # 2️⃣ SPLIT COMPOSITE FEATURES
        for composite, split_features in NORMALIZATION_RULES.items():
            if not isinstance(split_features, list):
                continue

            # deactivate composite
            cur.execute("""
                UPDATE features_master
                SET is_active = false
                WHERE name = %s
            """, (composite,))

            # insert atomic features
            for feat in split_features:
                cur.execute("""
                    INSERT INTO features_master (name, category)
                    SELECT %s, category
                    FROM features_master
                    WHERE name = %s
                    LIMIT 1
                    ON CONFLICT (name, category) DO NOTHING
                """, (feat, composite))

        # 3️⃣ MERGE DUPLICATES
        for old_name, canonical in NORMALIZATION_RULES.items():
            if isinstance(canonical, list):
                continue

            # deactivate old
            cur.execute("""
                UPDATE features_master
                SET is_active = false
                WHERE name = %s
            """, (old_name,))

            # ensure canonical exists
            cur.execute("""
                INSERT INTO features_master (name, category)
                SELECT %s, category
                FROM features_master
                WHERE name = %s
                LIMIT 1
                ON CONFLICT (name, category) DO NOTHING
            """, (canonical, old_name))

        conn.commit()
        return {"status": "feature master normalized"}
   
    
    def get_variant_features(
    self,
    variant_id: str,
    version: int = 1,
    categories=None
):
        """
        Get all features for a variant, optionally filtered by categories
        """
        query = """
            SELECT 
                vf.id,
                vf.variant_id,
                vf.feature_id,
                vf.value,
                vf.original_name,
                vf.version,
                fm.name as feature_name,
                fm.category
            FROM variant_features vf
            JOIN features_master fm ON vf.feature_id = fm.id
            WHERE vf.variant_id = %s 
            AND vf.version = %s
            AND fm.is_active = true
        """
        
        params = [variant_id, version]
        
        if categories:
            placeholders = ','.join(['%s'] * len(categories))
            query += f" AND fm.category IN ({placeholders})"
            params.extend(categories)
        
        query += " ORDER BY fm.category, fm.name"
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, tuple(params))
            results = cur.fetchall()
            return [dict(row) for row in results]
        
    def get_features(self, category=None):
        query = """
        SELECT id, name, category
        FROM features_master
        WHERE is_active = true
        """
        params = []

        if category:
            query += " AND category = %s"
            params.append(category)

        query += " ORDER BY category, name;"

        with self.get_conn().cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "category": r[2]
            }
            for r in rows
        ]
    
    def get_variant_feature_latest(self, variant_id: str, feature_id: str):
        query = """
        SELECT value, version
        FROM variant_features
        WHERE variant_id = %s
        AND feature_id = %s
        AND is_latest = true
        LIMIT 1;
        """

        with self.get_conn().cursor() as cur:
            cur.execute(query, (variant_id, feature_id))
            row = cur.fetchone()

        if not row:
            return None

        return {
            "value": row[0],
            "version": row[1]
        }
    

    def get_all_features_for_variant(self, variant_id: str):
        query = """
        SELECT 
            fm.id,
            fm.name,
            fm.category,
            vf.value
        FROM features_master fm
        LEFT JOIN variant_features vf
            ON vf.feature_id = fm.id
            AND vf.variant_id = %s
            AND vf.is_latest = true
        WHERE fm.is_active = true
        ORDER BY fm.category, fm.name;
        """

        with self.get_conn().cursor() as cur:
            cur.execute(query, (variant_id,))
            rows = cur.fetchall()

        return [
            {
                "feature_id": r[0],
                "feature_name": r[1],
                "category": r[2],
                "value": r[3]
            }
            for r in rows
        ]

    def update_variant_feature_value(self, variant_id: str, feature_id: str, value: str):
        with self.get_conn().cursor() as cur:

            # 1️⃣ Try update first
            cur.execute("""
                UPDATE variant_features
                SET value = %s
                WHERE variant_id = %s
                AND feature_id = %s
                AND is_latest = TRUE;
            """, (value, variant_id, feature_id))

            # 2️⃣ If nothing updated → insert new
            if cur.rowcount == 0:
                cur.execute("""
                    INSERT INTO variant_features (
                        variant_id, feature_id, value, is_latest
                    )
                    VALUES (%s, %s, %s, TRUE);
                """, (variant_id, feature_id, value))

        self.conn.commit()

    
    def create_feature(self, name: str, category: str, is_active: bool = True):
        with self.get_conn().cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO features_master (
                        name,
                        category,
                        is_active
                    )
                    VALUES (%s, %s, %s)
                    RETURNING id, name, category, is_active, created_at;
                """, (name, category, is_active))

                row = cur.fetchone()
                self.conn.commit()

                return {
                    "id": row[0],
                    "name": row[1],
                    "category": row[2],
                    "is_active": row[3],
                    "created_at": row[4]
                }

            except errors.UniqueViolation:
                self.conn.rollback()
                raise Exception("Feature with this name already exists in this category")

            except Exception as e:
                self.conn.rollback()
                raise e

    def get_categories(self):
        query = """
        SELECT DISTINCT category
        FROM features_master
        WHERE is_active = true
        AND category IS NOT NULL
        ORDER BY category;
        """

        with self.get_conn().cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

        return {
            "categories": [r[0] for r in rows]
        }

    def update_variant_feature(self, variant_id, feature_id, value, version=1):

        query = """
            UPDATE variant_features
            SET value = %s
            WHERE variant_id = %s
            AND feature_id = %s
            AND version = %s
        """

        with self.get_conn().cursor() as cur:
            cur.execute(query, (
                value,
                variant_id,
                feature_id,
                version
            ))

            if cur.rowcount == 0:
                return False   # No row updated

            self.get_conn().commit()
            return True




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
                conn.commit()
        return {
            "plan_id": str(r[0]),
            "name": r[1],
            "base_variant_class": r[2],
            "base_car_id": str(r[3]),
            "created_at": r[4].isoformat()
        }

    def get_plan_by_id(self, plan_id: str):
        query = """
            SELECT id, name, base_variant_class, base_car_id, created_at, updated_at
            FROM model_plans
            WHERE id = %s;
        """
        with self.get_conn().cursor() as cur:
            cur.execute(query, (plan_id,))
            r = cur.fetchone()
        if not r:
            return None
        return {
            "plan_id": str(r[0]),
            "name": r[1],
            "base_variant_class": r[2],
            "base_car_id": str(r[3]),
            "created_at": r[4].isoformat(),
            "updated_at": r[5].isoformat()
        }

    def list_plans(self, base_variant_class: str = None):
        if base_variant_class:
            query = """
                SELECT id, name, base_variant_class, base_car_id, created_at
                FROM model_plans
                WHERE base_variant_class = %s
                ORDER BY created_at DESC;
            """
            params = (base_variant_class,)
        else:
            query = """
                SELECT id, name, base_variant_class, base_car_id, created_at
                FROM model_plans
                ORDER BY created_at DESC;
            """
            params = ()

        with self.get_conn().cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            {
                "plan_id": str(r[0]),
                "name": r[1],
                "base_variant_class": r[2],
                "base_car_id": str(r[3]),
                "created_at": r[4].isoformat()
            }
            for r in rows
        ]

    def delete_plan(self, plan_id: str):
        query = """
            DELETE FROM model_plans
            WHERE id = %s
            RETURNING name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id,))
                r = cur.fetchone()
                conn.commit()
        return r[0] if r else None


class PlanFeatureDbManager(DbManager):
    def __init__(self):
        super().__init__()

    def bulk_insert_inherited_features(self, plan_id: str, features: list):
        """
        features: list of { feature_id, feature_name, category, value }
        Called once when plan is created to copy base class features.
        """
        query = """
            INSERT INTO plan_features
                (plan_id, feature_id, feature_name, category, value, original_value, is_inherited, cost_delta)
            VALUES (%s, %s, %s, %s, %s, %s, true, 0);
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, [
                    (plan_id, f["feature_id"], f["feature_name"], f["category"], f.get("value", ""), f.get("value", ""))
                    for f in features
                ])
                conn.commit()
        return len(features)

    def get_features_by_plan(self, plan_id: str, include_deleted: bool = False):
        if include_deleted:
            query = """
                SELECT id, feature_id, feature_name, category, value, original_value,
                       is_inherited, is_deleted, cost_delta
                FROM plan_features
                WHERE plan_id = %s
                ORDER BY category, feature_name;
            """
        else:
            query = """
                SELECT id, feature_id, feature_name, category, value, original_value,
                       is_inherited, is_deleted, cost_delta
                FROM plan_features
                WHERE plan_id = %s AND is_deleted = false
                ORDER BY category, feature_name;
            """
        with self.get_conn().cursor() as cur:
            cur.execute(query, (plan_id,))
            rows = cur.fetchall()

        return [
            {
                "plan_feature_id": str(r[0]),
                "feature_id": str(r[1]) if r[1] else None,
                "feature_name": r[2],
                "category": r[3],
                "value": r[4],
                "original_value": r[5],
                "is_inherited": r[6],
                "is_deleted": r[7],
                "cost_delta": float(r[8] or 0)
            }
            for r in rows
        ]

    def add_custom_feature(self, plan_id: str, feature_name: str, category: str,
                           value: str = None, cost_delta: float = 0):
        query = """
            INSERT INTO plan_features
                (plan_id, feature_id, feature_name, category, value, original_value, is_inherited, cost_delta)
            VALUES (%s, NULL, %s, %s, %s, %s, false, %s)
            RETURNING id, feature_name, category, value, original_value, cost_delta;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id, feature_name, category, value, value, cost_delta))
                r = cur.fetchone()
                conn.commit()
        return {
            "plan_feature_id": str(r[0]),
            "feature_name": r[1],
            "category": r[2],
            "value": r[3],
            "original_value": r[4],
            "cost_delta": float(r[5] or 0),
            "is_inherited": False
        }

    def update_feature(self, plan_id: str, plan_feature_id: str,
                       value: str = None, cost_delta: float = None):
        fields = []
        values = []

        if value is not None:
            fields.append("value = %s")
            values.append(value)
        if cost_delta is not None:
            fields.append("cost_delta = %s")
            values.append(cost_delta)

        if not fields:
            return None

        fields.append("updated_at = now()")
        values.extend([plan_id, plan_feature_id])

        query = f"""
            UPDATE plan_features
            SET {', '.join(fields)}
            WHERE plan_id = %s AND id = %s AND is_deleted = false
            RETURNING id, feature_name, value, cost_delta;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
                r = cur.fetchone()
                conn.commit()

        if not r:
            return None
        return {
            "plan_feature_id": str(r[0]),
            "feature_name": r[1],
            "value": r[2],
            "cost_delta": float(r[3] or 0)
        }

    def soft_delete_feature(self, plan_id: str, plan_feature_id: str):
        query = """
            UPDATE plan_features
            SET is_deleted = true, updated_at = now()
            WHERE plan_id = %s AND id = %s
            RETURNING id, feature_name;
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (plan_id, plan_feature_id))
                r = cur.fetchone()
                conn.commit()
        return {"plan_feature_id": str(r[0]), "feature_name": r[1]} if r else None

    def get_delta_summary(self, plan_id: str):
        query = """
            SELECT feature_name, cost_delta
            FROM plan_features
            WHERE plan_id = %s AND is_deleted = false AND cost_delta != 0
            ORDER BY cost_delta DESC;
        """
        with self.get_conn().cursor() as cur:
            cur.execute(query, (plan_id,))
            rows = cur.fetchall()

        breakdown = [
            {"feature_name": r[0], "cost_delta": float(r[1])}
            for r in rows
        ]
        total_delta = sum(b["cost_delta"] for b in breakdown)

        return {
            "total_delta": total_delta,
            "delta_direction": "increase" if total_delta > 0 else "decrease" if total_delta < 0 else "neutral",
            "breakdown": breakdown
        }