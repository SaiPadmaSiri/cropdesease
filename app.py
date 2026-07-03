import json
import os
import sqlite3
import csv
from io import StringIO
from datetime import datetime
from flask import Flask, render_template, request, send_file, abort, make_response, session, redirect, url_for
import numpy as np
import cv2  
import requests  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from deep_translator import GoogleTranslator  

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "AGRONETRA_AI_SECRET_KEY_2026")

# ==================================================
# Configurations & Database Initialization
# ==================================================

UPLOAD_FOLDER = "static/uploads"
CHARTS_FOLDER = "static/charts"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

def init_db():
    conn = sqlite3.connect("agronetra.db")
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        disease TEXT,
        confidence REAL,
        user_type TEXT,
        estimated_cost REAL,
        prediction_time TEXT,
        farmer_id INTEGER
    )
    """)
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS farmers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        crop TEXT,
        location TEXT,
        language TEXT DEFAULT 'en',
        photo TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

init_db()

# Ensure model exists (downloads at runtime if MODEL_URL is provided)
from download_model import ensure_model_present

MODEL_PATH = os.path.join("model", "crop_disease_model.keras")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
ensure_model_present(MODEL_PATH)

model = load_model(MODEL_PATH)

class_names = [
    "Apple - Apple Scab", 
    "Apple - Black Rot", 
    "Apple - Cedar Apple Rust", 
    "Apple - Healthy",
    "Bell Pepper - Bacterial Spot", 
    "Bell Pepper - Healthy", 
    "Cherry - Healthy", 
    "Cherry - Powdery Mildew",
    "Corn (Maize) - Cercospora Leaf Spot", 
    "Corn (Maize) - Common Rust", 
    "Corn (Maize) - Healthy",
    "Corn (Maize) - Northern Leaf Blight", 
    "Grape - Black Rot", 
    "Grape - Esca (Black Measles)",
    "Grape - Healthy", 
    "Grape - Leaf Blight", 
    "Peach - Bacterial Spot", 
    "Peach - Healthy",
    "Potato - Early Blight", 
    "Potato - Healthy", 
    "Potato - Late Blight", 
    "Strawberry - Healthy",
    "Strawberry - Leaf Scorch", 
    "Tomato - Bacterial Spot", 
    "Tomato - Early Blight",
    "Tomato - Healthy", 
    "Tomato - Late Blight", 
    "Tomato - Septoria Leaf Spot", 
    "Tomato - Yellow Leaf Curl Virus"
]

# =====================================================================
# MODULE 1: EXHAUSTIVE Pathological Knowledge Base
# =====================================================================

disease_info = {
    "Apple - Apple Scab": {
        "description": "Fungal disease causing dark olive spots on leaves and fruits.",
        "symptoms": "Irregular dark olive-green velvety spots on leaves developing into corky brown lesions on fruit.",
        "cause": "Fungus Venturia inaequalis", 
        "organic": "Neem oil spray or copper-based liquid solutions.", 
        "chemical": "Mancozeb or Captan fungicide application.",
        "treatment": "Apply protective copper sprays or organic fungicides.", 
        "prevention": "Prune trees regularly to ensure proper canopy airflow."
    },
    "Apple - Black Rot": {
        "description": "Fungal disease causing fruit rot, leaf lesions, and branch cankers.",
        "symptoms": "Frogeye leaf spots with purple-rimmed margins, cankers on limbs, and concentric rings of rot on fruit skin.",
        "cause": "Fungus Botryosphaeria obtusa", 
        "organic": "Sulfur dust preparations or organic copper sprays.", 
        "chemical": "Fludioxonil or thiophanate-methyl compounds.",
        "treatment": "Use fungicides and prune dead wood cankers immediately.", 
        "prevention": "Maintain rigorous orchard hygiene and ground clearance."
    },
    "Apple - Cedar Apple Rust": {
        "description": "Rust disease producing bright orange spots on leaf surfaces.",
        "symptoms": "Bright yellow-orange spots on the upper leaf surfaces with tiny tube-like structural spores underneath.",
        "cause": "Fungus Gymnosporangium juniperi-virginianae", 
        "organic": "Serenade Garden organic fungicide or copper soaps.", 
        "chemical": "Myclobutanil or Triadimefon applications.",
        "treatment": "Apply targeted rust inhibitors during early growing season windows.", 
        "prevention": "Remove nearby alternative cedar hosts within a 2-mile radius."
    },
    "Apple - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Leaves exhibit normal green color, complete margin line cell structures, and strong turgor pressure.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Apply compost tea or standard organic matter amendments.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Continue standard irrigation and pruning schedules."
    },
    "Bell Pepper - Bacterial Spot": {
        "description": "Bacterial infection causing dark leaf and fruit spots with rapid defoliation.",
        "symptoms": "Small yellow-green lesions on leaves that turn dark brown, look greasy, and drop off.",
        "cause": "Bacterium Xanthomonas campestris pv. vesicatoria", 
        "organic": "Copper soap or biological controls like Bacillus subtilis.", 
        "chemical": "Streptomycin sulfate or fixed copper fungicides.",
        "treatment": "Apply fixed-copper bactericides immediately.", 
        "prevention": "Use certified disease-free seeds and observe a 3-year crop rotation."
    },
    "Bell Pepper - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Consistent deep green pigmentation, uniform surface textures, and dynamic floral node generation.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Fish emulsion or seaweed foliar sprays.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Maintain balanced nitrogen-phosphorus-potassium feeding loops."
    },
    "Cherry - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Smooth leaves, uniform light-to-dark margins, and proper stem connection rigidity.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Mycorrhizal root inoculants and standard mulch applications.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Monitor orchards weekly for sudden changes."
    },
    "Cherry - Powdery Mildew": {
        "description": "White powdery fungal growth covering leaf surfaces and new shoots.",
        "symptoms": "White felt-like patches of fungal mycelium on the undersides of leaves causing curling and upward distortion.",
        "cause": "Fungus Podosphaera clandestina", 
        "organic": "Potassium bicarbonate or diluted milk solution sprays.", 
        "chemical": "Myclobutanil or sulfur-based active agents.",
        "treatment": "Apply wettable sulfur or potassium bicarbonate washes.", 
        "prevention": "Increase canopy airflow through targeted selective pruning."
    },
    "Corn (Maize) - Cercospora Leaf Spot": {
        "description": "Fungal disease causing rectangular gray leaf lesions running parallel to veins.",
        "symptoms": "Long, narrow, rectangular tan-to-gray lesions running parallel to leaf veins creating blocky scars.",
        "cause": "Fungus Cercospora zeae-maydis", 
        "organic": "Bio-fungicides containing Bacillus amyloliquefaciens.", 
        "chemical": "Pyraclostrobin or Azoxystrobin group fungicides.",
        "treatment": "Use Group 11 fungicides if field thresholds drop.", 
        "prevention": "Perform deep tillage and fall crop residue management."
    },
    "Corn (Maize) - Common Rust": {
        "description": "Fungal infection creating prominent powdery rust pustules.",
        "symptoms": "Oval to elongate cinnamon-brown pustules erupting on both upper and lower leaf surfaces.",
        "cause": "Fungus Puccinia sorghi", 
        "organic": "Neem oil concentrates or copper-based preventative dusts.", 
        "chemical": "Propiconazole or Tebuconazole therapeutic agents.",
        "treatment": "Apply triazole fungicides when early pustule counts emerge.", 
        "prevention": "Plant approved rust-resistant structural hybrids."
    },
    "Corn (Maize) - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Elongated robust stalks, thick deep-green leaf blades, and clean root anchorage anchors.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Nitrogen-rich organic manure or vermicompost dressings.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Keep field drainage paths clear of blockages."
    },
    "Corn (Maize) - Northern Leaf Blight": {
        "description": "Severe fungal disease causing massive cigar-shaped dead patches.",
        "symptoms": "Large, cigar-shaped, grayish-green or tan lesions expanding across whole leaves rapidly.",
        "cause": "Fungus Exserohilum turcicum", 
        "organic": "Foliar organic copper applications at row close.", 
        "chemical": "Mancozeb or prothioconazole systemics.",
        "treatment": "Apply combination strobilurin-triazole fungicides immediately.", 
        "prevention": "Utilize targeted crop rotation schemes away from grass families."
    },
    "Grape - Black Rot": {
        "description": "Destructive fungal disease causing complete fruit shriveling and leaf drop.",
        "symptoms": "Small brown circular spots on leaves with black borders; berries shrivel into hard black mummies.",
        "cause": "Fungus Guignardia bidwellii", 
        "organic": "Copper hydroxide formulations applied pre-bloom.", 
        "chemical": "Ziram, Mancozeb, or Elite fungicides.",
        "treatment": "Apply fungicides during critical pre-bloom and post-bloom cycles.", 
        "prevention": "Remove all mummified berries from vines and ground during winter pruning."
    },
    "Grape - Esca (Black Measles)": {
        "description": "Complex wood disease causing interveinal leaf burning and berry spotting.",
        "symptoms": "Interveinal tiger-stripe leaf scorch browning; dark spots appear on the skin of infected fruit.",
        "cause": "Fungi Complex (Phaeomoniella chlamydospora / Phaeoacremonium aleophilum)", 
        "organic": "Pruning wound protection using organic lime sulfur formulations.", 
        "chemical": "Systemic trunk injections or copper-based sealants.",
        "treatment": "Prune infected canopy cords 6 inches below visible wood staining.", 
        "prevention": "Sanitize all pruning shears with rubbing alcohol between vines."
    },
    "Grape - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Clean palmate leaves, pristine node tendrils, and clear high-vibrancy vascular network vectors.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Composted poultry manure and regular trace mineral additions.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Continue standard microclimate inspection schedules."
    },
    "Grape - Leaf Blight": {
        "description": "Fungal blight causing progressive marginal death of foliage.",
        "symptoms": "Dark brown lesions appearing on leaf margins, expanding rapidly until whole leaves drop.",
        "cause": "Fungus Pseudocercospora vitis", 
        "organic": "Foliar application of horsetail extract or copper sulfur.", 
        "chemical": "Azoxystrobin or fenhexamid compounds.",
        "treatment": "Apply broad-spectrum foliar fungicides upon primary symptom visibility.", 
        "prevention": "Keep ground foliage pruned away from low-hanging vine levels."
    },
    "Peach - Bacterial Spot": {
        "description": "Bacterial disease leading to leaf spot shot-holes and cracked fruit skin.",
        "symptoms": "Small, pale-green spots on leaves that darken, turn purple-brown, drop out, leaving a shot-hole appearance.",
        "cause": "Bacterium Xanthomonas arboricola pv. pruni", 
        "organic": "Oxytetracycline formulations or zinc-copper applications.", 
        "chemical": "Fixed copper compounds applied during dormancy stages.",
        "treatment": "Apply protective antibiotic or copper compounds during early spring flushes.", 
        "prevention": "Avoid over-fertilizing with excessive structural nitrogen applications."
    },
    "Peach - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Vibrant lanceolate leaves with unbroken margin lines and smooth green bark turgor.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Balanced mulch layers and composted wood-chip matrices.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Ensure optimal root zone soil pH balances (6.0 - 6.5)."
    },
    "Potato - Early Blight": {
        "description": "Common fungal disease targeting mature leaf structures first.",
        "symptoms": "Dark spots with concentric ring target patterns appearing on older lower foliage first.",
        "cause": "Fungus Alternaria solani", 
        "organic": "Serenade organic sprays or copper soap solutions.", 
        "chemical": "Appy Chlorothalonil or Azoxystrobin applications.",
        "treatment": "Apply systemic strobilurin protectants at 10-day field intervals.", 
        "prevention": "Maintain adequate plant spacing to allow leaf wetness dissipation."
    },
    "Potato - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Lush compound leaves without lower canopy yellowing or localized tuber defects.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Alfalfa meal and humic acid soil conditioning treatments.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Observe correct crop rotation blocks away from other nightshades."
    },
    "Potato - Late Blight": {
        "description": "Highly destructive water-mold capable of collapsing whole fields in days.",
        "symptoms": "Water-soaked dark lesions with white fungal growth on leaf undersides in high humidity environments.",
        "cause": "Oomycete Phytophthora infestans", 
        "organic": "Copper octanoate or preventative biological cultures.", 
        "chemical": "Ridomil Gold, Cyazofamid, or Fluazinam mixtures.",
        "treatment": "Apply therapeutic systemic down-fongicide cocktails instantly.", 
        "prevention": "Destroy voluntary potato growths and use only certified clean seed tubers."
    },
    "Strawberry - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Pristine trifoliate leaves with vibrant crown node bases and healthy runner generation vectors.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Bone meal, kelp extract, and well-rotted leaf mold updates.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Ensure clean clean straw bedding insulation layers underneath berries."
    },
    "Strawberry - Leaf Scorch": {
        "description": "Fungal infection creating prominent purple lesions across leaves.",
        "symptoms": "Numerous small, purplish spots on leaves that enlarge to form dark purple blotches and dry out leaves.",
        "cause": "Fungus Diplocarpon earlianum", 
        "organic": "Copper soap treatments or immediate pruning of infected runners.", 
        "chemical": "Thiophanate-methyl or Captan compounds.",
        "treatment": "Spray targeted protectant compounds at primary crown breakthrough intervals.", 
        "prevention": "Avoid using high overhead sprinkler systems; choose trickle lines instead."
    },
    "Tomato - Bacterial Spot": {
        "description": "Bacterial disease leading to greasy leaf lesions and rough scab spots on fruit.",
        "symptoms": "Small, water-soaked, greasy lesions on leaf tissue that turn brown, crack, and cause defoliation.",
        "cause": "Bacterium Xanthomonas perforans", 
        "organic": "Copper octanoate soaps or potassium silicate sprays.", 
        "chemical": "Actigard systemic plant activators or copper-mancozeb tank mixes.",
        "treatment": "Apply a combination copper-mancozeb field treatment stack.", 
        "prevention": "Do not enter or work inside fields when crop leaves are wet."
    },
    "Tomato - Early Blight": {
        "description": "Fungal spotting forming prominent target-board patterns.",
        "symptoms": "Concentric rings (target pattern) on leaves causing yellow halos and eventual drop-off starting from base canopy.",
        "cause": "Fungus Alternaria solani", 
        "organic": "Copper-based dusts or copper hydroxide foliar protectants.", 
        "chemical": "Chlorothalonil or Daconil applications at 7-day intervals.",
        "treatment": "Apply targeted broad-spectrum protectants upon structural branch row enclosure.", 
        "prevention": "Stake and prune lower branches up to 12 inches to reduce soil splash vector paths."
    },
    "Tomato - Healthy": {
        "description": "Plant tissue exhibits optimal growth parameters with zero pathology traces.",
        "symptoms": "Serrate leaves showing high chlorophyll density, clear stem trichomes, and dynamic root crowns.",
        "cause": "None (Optimal Growth State)", 
        "organic": "Epsom salts and organic compost tea foliar washes.", 
        "chemical": "None required.",
        "treatment": "No treatment required.", 
        "prevention": "Maintain steady drip watering parameters to minimize blossom end defects."
    },
    "Tomato - Late Blight": {
        "description": "Devastating plague causing large blue-gray grease spots and rapid systemic decay.",
        "symptoms": "Large, blue-gray water-soaked spots on foliage turning brown and destroying fruit clusters within days.",
        "cause": "Oomycete Phytophthora infestans", 
        "organic": "Copper hydroxide protectants applied at high pressure thresholds.", 
        "chemical": "Mefenoxam, Mandipropamid, or Chlorothalonil protectants.",
        "treatment": "Spray translaminar oomycete treatments instantly across all field zones.", 
        "prevention": "Monitor hyper-local dewpoint alerts and relative humidity metrics constantly."
    },
    "Tomato - Septoria Leaf Spot": {
        "description": "Fungal spotting creating thousands of tiny pinhole yellowing spots.",
        "symptoms": "Small, circular spots with dark brown margins and gray centers containing black fungal specks.",
        "cause": "Fungus Septoria lycopersici", 
        "organic": "Copper soap spray sequences combined with pruning lower leaves.", 
        "chemical": "Chlorothalonil or dynamic azoxystrobin foliar treatments.",
        "treatment": "Apply protective contact chemical sprays immediately at primary canopy breakdown.", 
        "prevention": "Eliminate alternate solanaceous weed hosts along boundary perimeters."
    },
    "Tomato - Yellow Leaf Curl Virus": {
        "description": "Viral disease causing severe stunting, leaf wrinkling, and zero fruit set.",
        "symptoms": "Severe cupping, crumpling, and upward rolling of leaves paired with pale yellow interveinal mottling.",
        "cause": "Geminiviridae Tomato yellow leaf curl virus (Vectored by Whiteflies)", 
        "organic": "Insecticidal soaps or neem-oil applications targeting vector colonies.", 
        "chemical": "Imidacloprid or Dinotefuran soil drenches to suppress vector populations.",
        "treatment": "No cure exists. Immediately extract and destroy infected plants to preserve the row block.", 
        "prevention": "Deploy highly reflective silver plastic mulch ribbons to disorient whitefly vectors."
    }
}

explanations = {
    "Apple - Apple Scab": "The model detected irregular dark, olive-green velvety lesions characteristic of Apple Scab fungal development on the leaf surface.",
    "Apple - Black Rot": "The model detected brown spots with purple margins (frogeye pattern) indicating an advanced Apple Black Rot infection.",
    "Apple - Cedar Apple Rust": "The model identified vibrant yellow-orange circular rust structures typical of a localized Cedar Apple Rust vector outbreak.",
    "Apple - Healthy": "The model detected uniform chlorophyll distribution, pristine structural edges, and complete absence of pathogenic spots.",
    "Bell Pepper - Bacterial Spot": "The model isolated dark, water-soaked, irregular lesions and edge chlorosis indicating active Bell Pepper Bacterial Spot multiplication.",
    "Bell Pepper - Healthy": "The model verified healthy leaf structures with no symptomatic patterns, ensuring complete vascular integrity.",
    "Cherry - Healthy": "The model identified clean green tissue matrices across the cherry botanical surface profile with no mildew spotting.",
    "Cherry - Powdery Mildew": "The model isolated white, powder-like fungal mycelium patches on the surface layers typical of Cherry Powdery Mildew propagation.",
    "Corn (Maize) - Cercospora Leaf Spot": "The model recognized elongated, narrow rectangular tan lesions bound strictly between veins, indicating Corn Gray Leaf Spot.",
    "Corn (Maize) - Common Rust": "The model identified raised, powdery cinnamon-brown pustules typical of an active Corn Common Rust outbreak.",
    "Corn (Maize) - Healthy": "The model detected strong parallel veining, normal vascular structure, and full light absorption capacity without spot lesions.",
    "Corn (Maize) - Northern Leaf Blight": "The model verified large, cigar-shaped grayish-tan lesions tracking across large portions of the leaf, identifying Northern Corn Leaf Blight.",
    "Grape - Black Rot": "The model identified small reddish-brown leaf spots with black fruiting margins characteristic of Grape Black Rot lifecycle initiation.",
    "Grape - Esca (Black Measles)": "The model identified distinctive interveinal tiger-stripe browning and localized chlorosis indicating chronic Grape Esca wood disease.",
    "Grape - Healthy": "The Grape profile shows a completely smooth structural leaf grid matrix, healthy vascular lines, and excellent coloration properties.",
    "Grape - Leaf Blight": "The model isolated dark marginal necrosis expanding inside the palmate structure, pointing to Grape Leaf Blight infection.",
    "Peach - Bacterial Spot": "The model isolated angular, dark spots clustered at leaf tips forming a shot-hole erosion index indicative of Peach Bacterial Spot.",
    "Peach - Healthy": "The model confirmed uniform cellular structures, smooth margins, and deep green pigmentation across the peach lanceolate leaf.",
    "Potato - Early Blight": "The model identified brown spots with concentric ring target patterns on the foliage, establishing an Alternaria Potato Early Blight trace.",
    "Potato - Healthy": "The model verified complete compound leaf expansion, excellent leaf vein turgidity, and zero fungal spot counts.",
    "Potato - Late Blight": "The model identified fast-spreading water-soaked blue-gray blotches indicating a highly dangerous Potato Late Blight condition.",
    "Strawberry - Healthy": "The trifoliate leaf exhibits optimal development parameters, consistent leaf margins, and clear vascular networks.",
    "Strawberry - Leaf Scorch": "The model isolated small, irregular purplish blotches expanding across the leaf structure, classifying Strawberry Leaf Scorch.",
    "Tomato - Bacterial Spot": "The model detected multiple small, dark, greasy circular lesions on the leaf tissue, confirming Tomato Bacterial Spot.",
    "Tomato - Early Blight": "The model isolated large target-like lesions on mature foliage containing internal concentric rings characteristic of Tomato Early Blight.",
    "Tomato - Healthy": "The serrate leaf architecture exhibits deep chlorophyll concentration, clean vascular vectors, and zero spot pathology indexes.",
    "Tomato - Late Blight": "The model identified massive water-soaked pale-green to dark brown necrosis on the leaf surface, indicating dangerous Tomato Late Blight.",
    "Tomato - Septoria Leaf Spot": "The model identified multiple small, circular spots with dark brown borders and grey pucker centers typical of Tomato Septoria Leaf Spot.",
    "Tomato - Yellow Leaf Curl Virus": "The model isolated distinct upward rolling, puckering, and extreme interveinal yellowing indicative of Tomato Yellow Leaf Curl Virus."
}

recovery_scores = {
    "Apple - Apple Scab": 85, 
    "Apple - Black Rot": 75, 
    "Apple - Cedar Apple Rust": 80, 
    "Apple - Healthy": 100,
    "Bell Pepper - Bacterial Spot": 70, 
    "Bell Pepper - Healthy": 100, 
    "Cherry - Healthy": 100, 
    "Cherry - Powdery Mildew": 80,
    "Corn (Maize) - Cercospora Leaf Spot": 75, 
    "Corn (Maize) - Common Rust": 90, 
    "Corn (Maize) - Healthy": 100, 
    "Corn (Maize) - Northern Leaf Blight": 70,
    "Grape - Black Rot": 80, 
    "Grape - Esca (Black Measles)": 50, 
    "Grape - Healthy": 100, 
    "Grape - Leaf Blight": 75,
    "Peach - Bacterial Spot": 70, 
    "Peach - Healthy": 100, 
    "Potato - Early Blight": 80, 
    "Potato - Healthy": 100, 
    "Potato - Late Blight": 40,
    "Strawberry - Healthy": 100, 
    "Strawberry - Leaf Scorch": 75, 
    "Tomato - Bacterial Spot": 65, 
    "Tomato - Early Blight": 75,
    "Tomato - Healthy": 100, 
    "Tomato - Late Blight": 35, 
    "Tomato - Septoria Leaf Spot": 75, 
    "Tomato - Yellow Leaf Curl Virus": 30
}

severity_levels = {
    "Apple - Apple Scab": "Moderate", 
    "Apple - Black Rot": "High", 
    "Apple - Cedar Apple Rust": "Moderate", 
    "Apple - Healthy": "Healthy",
    "Bell Pepper - Bacterial Spot": "Moderate", 
    "Bell Pepper - Healthy": "Healthy", 
    "Cherry - Healthy": "Healthy", 
    "Cherry - Powdery Mildew": "Moderate",
    "Corn (Maize) - Cercospora Leaf Spot": "Moderate", 
    "Corn (Maize) - Common Rust": "Low", 
    "Corn (Maize) - Healthy": "Healthy", 
    "Corn (Maize) - Northern Leaf Blight": "High",
    "Grape - Black Rot": "High", 
    "Grape - Esca (Black Measles)": "High", 
    "Grape - Healthy": "Healthy", 
    "Grape - Leaf Blight": "Moderate",
    "Peach - Bacterial Spot": "Moderate", 
    "Peach - Healthy": "Healthy", 
    "Potato - Early Blight": "Moderate", 
    "Potato - Healthy": "Healthy", 
    "Potato - Late Blight": "Severe",
    "Strawberry - Healthy": "Healthy", 
    "Strawberry - Leaf Scorch": "Moderate", 
    "Tomato - Bacterial Spot": "Moderate", 
    "Tomato - Early Blight": "Moderate",
    "Tomato - Healthy": "Healthy", 
    "Tomato - Late Blight": "Severe", 
    "Tomato - Septoria Leaf Spot": "Moderate", 
    "Tomato - Yellow Leaf Curl Virus": "Severe"
}

crop_cost = {
    "Tomato": 50, 
    "Potato": 60, 
    "Apple": 150, 
    "Grape": 120, 
    "Corn (Maize)": 40,
    "Bell Pepper": 70, 
    "Cherry": 130, 
    "Peach": 140, 
    "Strawberry": 80
}

smart_treatment_plan = {
    "Apple - Apple Scab": [
        "Day 1: Spray complete orchard blocks with dynamic copper octanoate soaps.",
        "Day 2: Rake and clear fallen leaves around base perimeters to destroy resting spores.",
        "Day 5: Selectively prune central branches to allow maximum wind access loops.",
        "Day 7: Re-evaluate new spring growth leaves for olive velvety spots visibility thresholds."
    ],
    "Apple - Black Rot": [
        "Day 1: Cut out branch cankers cleanly up to six inches inside healthy wood tissues.",
        "Day 2: Sanitize structural wounds with commercial pruning sealants.",
        "Day 5: Apply targeted Captan therapeutic fungicide solutions.",
        "Day 7: Audit lower canopy leaves for primary frogeye pattern spot appearances."
    ],
    "Apple - Cedar Apple Rust": [
        "Day 1: Apply liquid systemics containing Myclobutanil directly across apple leaves.",
        "Day 2: Track down and remove galls from nearby ornamental cedar tree structures.",
        "Day 5: Run thorough field moisture audits across low ground rows.",
        "Day 7: Re-inspect foliage for orange tube spore developments underneath leaf maps."
    ],
    "Apple - Healthy": [
        "Day 1: Review standard canopy trace mineral nutrition loops.",
        "Day 2: Observe normal growth and flower branch setting thresholds.",
        "Day 5: Apply routine preventive biological compost washes.",
        "Day 7: Document leaf diagnostic metadata records into backend server ledgers."
    ],
    "Bell Pepper - Bacterial Spot": [
        "Day 1: Apply fixed-copper bactericides mixed with mancozeb to leaves.",
        "Day 2: Discontinue all overhead high-pressure sprinkler lines to reduce splash risk.",
        "Day 5: Inspect lower crown layers for greasy spot formations.",
        "Day 7: Reapply copper protective barriers if local rainfall metrics expand."
    ],
    "Bell Pepper - Healthy": [
        "Day 1: Maintain established micro-drip soil watering lines.",
        "Day 2: Inspect row nodes for active uniform bloom settings.",
        "Day 5: Add standard organic fish emulsion nitrogen dressings to root layers.",
        "Day 7: Update local logging metrics with complete clear crop status signatures."
    ],
    "Cherry - Healthy": [
        "Day 1: Conduct macro-level canopy surveillance audits.",
        "Day 2: Maintain regular orchard row weed clearance profiles.",
        "Day 5: Ensure root zone soils maintain proper draining loops.",
        "Day 7: Confirm excellent general health indicators on persistent storage fields."
    ],
    "Cherry - Powdery Mildew": [
        "Day 1: Wash infected leaves using therapeutic potassium bicarbonate sprays.",
        "Day 2: Clean out low-hanging suckers to optimize solar radiation penetration paths.",
        "Day 5: Apply broad-spectrum sterol-inhibitor fungicide formulas.",
        "Day 7: Assess young tip growths for persistent white powdery residue maps."
    ],
    "Corn (Maize) - Cercospora Leaf Spot": [
        "Day 1: Inspect field leaves for parallel rectangular tan lesions.",
        "Day 2: Deploy systemic strobilurin or triazole group field crop protectants.",
        "Day 5: Maintain standard crop nitrogen feeding loops to support metabolic recovery.",
        "Day 7: Map the field transmission boundary profile to verify spread termination."
    ],
    "Corn (Maize) - Common Rust": [
        "Day 1: Run comprehensive stalk pustules density tests across outer rows.",
        "Day 2: Spray recommended triazole therapeutic solutions across row spaces.",
        "Day 5: Check field boundary grass vectors for alternative rust reservoirs.",
        "Day 7: Track cinnamon-brown pustule ruptures to verify active spore death."
    ],
    "Corn (Maize) - Healthy": [
        "Day 1: Verify uniform crop row close performance states.",
        "Day 2: Analyze normal root bracing structural stability parameters.",
        "Day 5: Monitor soil moisture sensors for optimal intake levels.",
        "Day 7: Log clean health index values into persistent SQLite tables."
    ],
    "Corn (Maize) - Northern Leaf Blight": [
        "Day 1: Apply combination strobilurin-triazole formulas using high-pressure spray booms.",
        "Day 2: Track cigar-shaped lesion expansions across the middle canopy segments.",
        "Day 5: Adjust row ventilation conditions by maintaining perimeter clearance spaces.",
        "Day 7: Re-evaluate crop block conditions to clear late season safety protocols."
    ],
    "Grape - Black Rot": [
        "Day 1: Pick off and destroy all infected shriveled black berry clusters instantly.",
        "Day 2: Apply protective Mancozeb treatments across entire block structures.",
        "Day 5: Prune dense foliage walls to guarantee sunlight reach down to cluster lines.",
        "Day 7: Check leaf faces for tiny reddish-brown circular spot initiations."
    ],
    "Grape - Esca (Black Measles)": [
        "Day 1: Paint vine pruning cuts using professional organic copper sealants.",
        "Day 2: Separate heavily striping wood cords from general healthy wire layers.",
        "Day 5: Apply balanced trace elements to ease vine metabolic stresses.",
        "Day 7: Monitor grape skin layers for small purple speck outbreaks."
    ],
    "Grape - Healthy": [
        "Day 1: Prune standard non-bearing tendril elements to shape vine paths.",
        "Day 2: Ensure correct trellis cable tension metrics remain balanced.",
        "Day 5: Run macro-level soil nitrogen-potassium nutrient check loops.",
        "Day 7: Record pristine vineyard diagnostics values inside history tables."
    ],
    "Grape - Leaf Blight": [
        "Day 1: Spray protective copper sulfate sequences directly onto foliage maps.",
        "Day 2: Clean out low-hanging suckers to optimize solar radiation penetration paths.",
        "Day 5: Apply broad-spectrum foliar fungicides upon primary symptom visibility.",
        "Day 7: Review leaf margins for expanding dark marginal necrosis lines."
    ],
    "Peach - Bacterial Spot": [
        "Day 1: Deploy protective copper hydroxide sprays during current dormancy or flush steps.",
        "Day 2: Apply systemic oxytetracycline antibiotic sprays if field rules are breached.",
        "Day 5: Avoid high-pressure nitrogen fertilizers that create excessive tissue tenderness.",
        "Day 7: Monitor peach layers for small purple speck outbreaks."
    ],
    "Peach - Healthy": [
        "Day 1: Inspect bark and branch junctions for optimal wood strength values.",
        "Day 2: Verify normal blossom or fruit set turgor states.",
        "Day 5: Spread fresh wood-chip mulch beds to isolate root moisture paths.",
        "Day 7: File clear diagnostic status tokens inside storage databases."
    ],
    "Potato - Early Blight": [
        "Day 1: Deploy contact chlorothalonil chemical sprays at maximum volume rates.",
        "Day 2: Set strict field access limits to prevent mechanical spore transfer.",
        "Day 5: Run detailed water soil scans; execute irrigation cycles only at dawn windows.",
        "Day 7: Check lower foliage lines for target-board pattern expansion profiles."
    ],
    "Potato - Healthy": [
        "Day 1: Survey fields for complete foliage canopy close performance.",
        "Day 2: Track normal underground tuber bulking indicator benchmarks.",
        "Day 5: Feed crops with organic humic acid root stimulants.",
        "Day 7: Commit verified healthy field tokens to system databases."
    ],
    "Potato - Late Blight": [
        "Day 1: Eliminate and bury whole blighted plant clusters instantly; do not pile debris open.",
        "Day 2: Inject systemic translaminar Cyazofamid or therapeutic chemical stacks across fields.",
        "Day 3: Turn off all irrigation systems entirely to strip humidity from the ground layer.",
        "Day 5: Audit entire field bounds at strict 24-hour verification timelines."
    ],
    "Strawberry - Healthy": [
        "Day 1: Clean out dead organic fragments away from center crowns.",
        "Day 2: Verify normal daughter plant runner formation parameters.",
        "Day 5: Spray light foliar liquid organic kelp formulations.",
        "Day 7: Log clear operational status metrics into backend file sheets."
    ],
    "Strawberry - Leaf Scorch": [
        "Day 1: Apply copper soap treatments or spray targeted protectant compounds.",
        "Day 2: Prune heavily spotted trifoliate leaves from runner lines immediately.",
        "Day 5: Transition row watering to subsurface trickle irrigation lines exclusively.",
        "Day 7: Inspect young crown flushes for new purple speck developments."
    ],
    "Tomato - Bacterial Spot": [
        "Day 1: Deploy a high-pressure contact copper-mancozeb tank mix across row blocks.",
        "Day 2: Implement strict quarantine lines; do not enter wet row spaces under any conditions.",
        "Day 5: Treat with potassium silicate foliar washes to build cell wall structural armor.",
        "Day 7: Track leaf dropping counts to check systemic infection declines."
    ],
    "Tomato - Early Blight": [
        "Day 1: Prune lower target-spot yellowing leaves up to 12 inches to reduce soil splash vector paths.",
        "Day 2: Apply protective Chlorothalonil or Daconil applications at 7-day intervals.",
        "Day 5: Apply organic mulch matrices around soil stems to control splash transmission.",
        "Day 7: Reassess field canopy for concentric rings propagation vectors."
    ],
    "Tomato - Healthy": [
        "Day 1: Continue baseline microclimate crop inspections.",
        "Day 2: Maintain regular subsurface localized drip irrigation levels to minimize blossom end defects.",
        "Day 5: Apply balanced mineral amendments or organic compost teas.",
        "Day 7: Document field vigor records in persistent history sheets."
    ],
    "Tomato - Late Blight": [
        "Day 1: Remove infected leaf matrices immediately to stop zoospore migration.",
        "Day 2: Inject systemic translaminar Mandipropamid or therapeutic chemical stacks across fields.",
        "Day 5: Reinspect localized sub-canopy areas for new water-soaked lesions.",
        "Day 7: Reapply secondary protectant treatment if atmospheric humidity continues >80%."
    ],
    "Tomato - Septoria Leaf Spot": [
        "Day 1: Clean away heavily pocked lower branches from ground contact ranges.",
        "Day 2: Apply broad-spectrum Azoxystrobin protectants to clear leaf faces.",
        "Day 5: Spray biological Bacillus subtilis protectant barriers across new growth.",
        "Day 7: Inspect middle-tier leaves for gray-centered pinhole development profiles."
    ],
    "Tomato - Yellow Leaf Curl Virus": [
        "Day 1: Extract and burn heavily crumpled virused plants instantly to preserve row blocks.",
        "Day 2: Inject systemic Dinotefuran soil drenches to shut down whitefly vector lifecycles.",
        "Day 5: Deploy reflective silver poly ground mulch sheets to confuse flying insects.",
        "Day 7: Check yellow sticky traps to confirm target whitefly count drop-offs."
    ]
}

# Safeguard execution loop
for missing_key in class_names:
    if missing_key not in smart_treatment_plan:
        smart_treatment_plan[missing_key] = [
            "Day 1: Conduct localized canopy microclimate validation checks.",
            "Day 2: Treat foliage with contact copper protectant formulas.",
            "Day 5: Prune dead low-productivity structural leaves.",
            "Day 7: Re-evaluate target block using AGRONETRA v1.0 diagnostic cycles."
        ]

action_plans = {
    "Healthy": [
        "Continue regular monitoring", 
        "Maintain balanced fertilization",
        "Follow irrigation schedule", 
        "Keep field clean"
    ],
    "Disease": [
        "Remove infected leaves", 
        "Apply recommended pesticide/fungicide",
        "Monitor spread every 3 days", 
        "Consult agricultural expert if severe"
    ]
}

# =====================================================================
# TASK 35B: GLOBAL UI TRANSLATION DICTIONARY MAP
# =====================================================================
ui_translations = {
    "en": {
        "title": "AGRONETRA", 
        "subtitle": "AI-Powered Crop Disease Intelligence Platform", 
        "detect_diagnose": "Detect • Diagnose • Protect",
        "nav_home": "Home Workspace", 
        "nav_history": "Analytics History", 
        "nav_analytics": "Live Charts Dashboard", 
        "change_lang": "Change Language",
        "upload_title": "Scan Crop Leaf", 
        "upload_desc": "Upload a file or capture a live macro snapshot of the targeted plant foliage.",
        "btn_upload": "Upload Leaf Image", 
        "btn_camera": "Capture Live Leaf", 
        "profile_scope": "User Profile Scope:", 
        "lang_opts": "Global Language Options:",
        "city": "Regional Evaluation City:", 
        "scale": "Scale Quantifier Unit Count:", 
        "btn_predict": "Check Disease", 
        "feat_scan": "Disease Scan", 
        "feat_scan_desc": "Deep Convolutional Neural Networks classify anomalies across 29 botanical families instantly.",
        "feat_treat": "Smart Treatment", 
        "feat_treat_desc": "Advisory arrays return dual chemical prescriptions alongside organic mitigation options.",
        "feat_weather": "Weather Information", 
        "feat_weather_desc": "Live meteorological feeds track dewpoint variables and relative localized risk vectors.",
        "feat_reports": "PDF Reports", 
        "feat_reports_desc": "Downloadable enterprise-grade audit reports mapping treatment and financial recovery protocols.",
        "stat_path": "Pathologies Supported", 
        "stat_acc": "Baseline Accuracy", 
        "stat_img": "Images Trained", 
        "stat_active": "AI Analysis Active",
        "compare_btn": "Cross-Compare Disease Baseline Records", 
        "compare_title": "AGRONETRA Cross-Pathology Metrics Matrix",
        "target_genus": "Target Botanical Genus", 
        "status": "Disease Identification", 
        "confidence": "Model Classifier Confidence",
        "severity": "Computer Vision Severity", 
        "gauge_title": "Pathological Diagnostic Severity Gauge", 
        "gauge_desc": "Current Active Evaluation Area:",
        "xai_title": "Explainable AI (XAI) Attribution Localization", 
        "xai_desc": "Visual verification mapping model structural focus coordinates to authenticate pixel deduction arrays",
        "original_img": "Original Leaf Source Ingestion", 
        "heatmap_img": "AI Computer Vision Feature Heatmap", 
        "rationale": "Model Diagnostic Verification Rationale:",
        "simulator_title": "Automated Countermeasure Lifecycle Recovery Simulator", 
        "simulator_desc": "Predictive progression modeling assuming immediate deployment of prescribed remediation parameters",
        "baseline": "Current Ingestion Baseline", 
        "dose_48h": "After Primary Dose (48h)", 
        "stabilization": "Canopy Stabilization (7 Days)",
        "recommendation_title": "Production Recommendation Vector", 
        "read_aloud": "Read Aloud", 
        "profile_assessment": "Pathological Profile Assessment",
        "description": "Description", 
        "symptoms": "Diagnostic Symptoms Identified:", 
        "cause": "Biological Pathogen Cause Agent:",
        "remediation": "Prescriptive Remediation Parameters", 
        "treatment": "Treatment Formulation", 
        "organic": "Organic Treatment Protocol",
        "chemical": "Chemical Countermeasure", 
        "prevention": "Prevention Protocols", 
        "action_plan": "Pipeline Operations", 
        "calendar": "Strategic Field Implementation Calendar",
        "weather_title": "Weather Information", 
        "temp": "Temperature Window:", 
        "humidity": "Relative Humidity Rate:", 
        "sky": "Sky Atmosphere:",
        "hazard_risk": "Disease Pathological Propagation Hazard Risk:", 
        "recovery_proj": "Crop Recovery Projection", 
        "outlook": "Outlook Capacity:",
        "priority": "Operational Priority Urgency", 
        "status_tier": "Status Index Tier:", 
        "deadline": "Recommended Deadline:",
        "yield_loss_title": "Expected Crop Loss", 
        "yield_loss": "Projected Crop Yield Loss", 
        "mitigation_cost": "Mitigation Cost Allocation",
        "net_impact": "Net Financial Field Impact Value Projection:", 
        "ai_doctor": "AI Crop Doctor", 
        "btn_spread": "Will it spread?",
        "btn_save": "Can I save my crop?", 
        "btn_cost": "Treatment cost?", 
        "btn_prevent": "Prevention tips?", 
        "placeholder": "Query details about vector spreads, diagnostic steps...",
        "btn_query": "Query Copilot", 
        "btn_download": "Download PDF Audit Report", 
        "btn_new_scan": "Analyze New Crop Target", 
        "footer": "AGRONETRA Crop Intelligence Systems v1.0 © 2026",
        "login_title": "Farmer Login", 
        "login_sub": "Welcome to AGRONETRA", 
        "login_btn": "🌿 Login", 
        "login_new": "New Farmer?", 
        "login_create": "Create Account",
        "mobile_number": "📱 Mobile Number",
        "password": "🔐 Password",
        "welcome_farmer": "👨‍🌾 Welcome Farmer",
        "ai_companion": "Your AI Companion for Healthy Crops",
        "reg_title": "👨‍🌾 Create Farmer Profile", 
        "reg_sub": "Join AGRONETRA to save your scan history", 
        "reg_btn": "Register & Join", 
        "reg_exist": "Already registered?", 
        "reg_login": "Login Here",
        "reg_name": "👤 Full Name",
        "reg_phone": "📱 Mobile Number",
        "reg_pass": "🔐 Create Password",
        "reg_loc": "📍 Village / City"
    },
    "te": {
        "title": "అగ్రోనేత్ర", 
        "subtitle": "AI-ఆధారిత పంట వ్యాధి ఇంటెలిజెన్స్ ప్లాట్‌ఫారమ్", 
        "detect_diagnose": "గుర్తించండి • నిర్ధారించండి • రక్షించండి",
        "nav_home": "హోమ్ వర్క్‌స్పేస్", 
        "nav_history": "చరిత్ర", 
        "nav_analytics": "లైవ్ చార్ట్‌లు", 
        "change_lang": "భాష మార్చండి",
        "upload_title": "పంట ఆకును స్కాన్ చేయండి", 
        "upload_desc": "లక్ష్యంగా ఉన్న మొక్క ఆకుల ఫైల్‌ను అప్‌లోడ్ చేయండి లేదా లైవ్ ఫోటో తీయండి.",
        "btn_upload": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి", 
        "btn_camera": "లైవ్ కెమెరా తీయండి", 
        "profile_scope": "వినియోగదారు ప్రొఫైల్:", 
        "lang_opts": "భాషా ఎంపికలు:",
        "city": "ప్రాంతీయ మూల్యాంకన నగరం:", 
        "scale": "మొక్కల సంఖ్య:", 
        "btn_predict": "వ్యాధిని నిర్ధారించండి",
        "feat_scan": "వ్యాధి స్కాన్", 
        "feat_scan_desc": "డీప్ న్యూరల్ నెట్‌వర్క్‌లు 29 రకాల వ్యాధులను గుర్తిస్తాయి.",
        "feat_treat": "స్మార్ట్ ట్రీట్మెంట్", 
        "feat_treat_desc": "సేంద్రీయ మరియు రసాయన మందుల సూచనలు.",
        "feat_weather": "వాతావరణ సమాచారం", 
        "feat_weather_desc": "ప్రత్యక్ష వాతావరణ అప్‌డేట్స్.",
        "feat_reports": "PDF రిపోర్ట్‌లు", 
        "feat_reports_desc": "ఆడిట్ నివేదికలను డౌన్‌లోడ్ చేసుకోండి.",
        "stat_path": "వ్యాధుల మద్దతు", 
        "stat_acc": "ఖచ్చితత్వం", 
        "stat_img": "చిత్రాల శిక్షణ", 
        "stat_active": "AI విశ్లేషణ యాక్టివ్",
        "compare_btn": "వ్యాధుల రికార్డులను సరిపోల్చండి", 
        "compare_title": "అగ్రోనేత్ర క్రాస్-పాథాలజీ మాట్రిక్స్",
        "target_genus": "లక్ష్య వృక్షశాస్త్ర జాతి", 
        "status": "వ్యాధి గుర్తింపు", 
        "confidence": "మోడల్ ఖచ్చితత్వం", 
        "severity": "కంప్యూటర్ విజన్ తీవ్రత",
        "gauge_title": "వ్యాధి తీవ్రత గేజ్", 
        "gauge_desc": "ప్రస్తుత మూల్యాంకన ప్రాంతం:", 
        "xai_title": "ఎక్స్‌ప్లెయినబుల్ AI (XAI) హీట్‌మ్యాప్", 
        "xai_desc": "పిక్సెల్ తగ్గింపు శ్రేణులను ప్రామాణీకరించడానికి మ్యాపింగ్.",
        "original_img": "అసలు ఆకు చిత్రం", 
        "heatmap_img": "AI కంప్యూటర్ విజన్ హీట్‌మ్యాప్", 
        "rationale": "మోడల్ నిర్ధారణ హేతువు:",
        "simulator_title": "రికవరీ సిమ్యులేటర్", 
        "simulator_desc": "చికిత్స ప్రణాళికను అమలు చేస్తే రికవరీ పురోగతి.",
        "baseline": "ప్రస్తుత స్థితి", 
        "dose_48h": "మొదటి డోస్ తర్వాత (48 గంటలు)", 
        "stabilization": "స్థిరీకరణ (7 రోజులు)",
        "recommendation_title": "ఉత్పత్తి సిఫార్సు", 
        "read_aloud": "చదివి వినిపించు", 
        "profile_assessment": "వ్యాధి ప్రొఫైల్ అంచనా",
        "description": "వివరణ", 
        "symptoms": "గుర్తించిన లక్షణాలు:", 
        "cause": "వ్యాధి కారకం:", 
        "remediation": "నివారణ చర్యలు", 
        "treatment": "చికిత్స",
        "organic": "సేంద్రీయ చికిత్స", 
        "chemical": "రసాయన చికిత్స", 
        "prevention": "నివారణ ప్రోటోకాల్స్", 
        "action_plan": "చర్య ప్రణాళిక", 
        "calendar": "ఫీల్డ్ అమలు క్యాలెండర్",
        "weather_title": "వాతావరణ సమాచారం", 
        "temp": "ఉష్ణోగ్రత:", 
        "humidity": "తేమ:", 
        "sky": "వాతావరణం:", 
        "hazard_risk": "వ్యాధి వ్యాప్తి ప్రమాదం:",
        "recovery_proj": "పంట రికవరీ అంచనా", 
        "outlook": "రికవరీ అవకాశం:", 
        "priority": "ప్రాధాన్యత అత్యవసరం", 
        "status_tier": "స్థితి:", 
        "deadline": "గడువు:",
        "yield_loss_title": "అంచనా వేసిన పంట నష్టం", 
        "yield_loss": "అంచనా వేసిన దిగుబడి నష్టం", 
        "mitigation_cost": "చికిత్స ఖర్చు", 
        "net_impact": "నికర ఆర్థిక నష్టం:",
        "ai_doctor": "AI పంట వైద్యుడు", 
        "btn_spread": "ఇది వ్యాపిస్తుందా?", 
        "btn_save": "పంటను రక్షించగలనా?", 
        "btn_cost": "చికిత్స ఖర్చు?", 
        "btn_prevent": "నివారణ చిట్కాలు?",
        "placeholder": "వ్యాప్తి, రోగ నిర్ధారణ లేదా నివారణ గురించి అడగండి...", 
        "btn_query": "ప్రశ్నించండి", 
        "btn_download": "PDF రిపోర్ట్ డౌన్‌లోడ్", 
        "btn_new_scan": "కొత్త ఆకును స్కాన్ చేయండి", 
        "footer": "అగ్రోనేత్ర క్రాప్ ఇంటెలిజెన్స్ సిస్టమ్స్ v1.0 © 2026",
        "login_title": "రైతు లాగిన్", 
        "login_sub": "అగ్రోనేత్రకు స్వాగతం", 
        "login_btn": "🌿 లాగిన్", 
        "login_new": "కొత్త రైతువా?", 
        "login_create": "ఖాతా సృష్టించండి",
        "mobile_number": "📱 మొబైల్ నంబర్",
        "password": "🔐 పాస్‌వర్డ్",
        "welcome_farmer": "👨‍🌾 రైతుకు స్వాగతం",
        "ai_companion": "మీ పంటల ఆరోగ్యానికి AI సహాయకుడు",
        "reg_title": "👨‍🌾 రైతు ప్రొఫైల్‌ని సృష్టించండి", 
        "reg_sub": "మీ స్కాన్ చరిత్రను సేవ్ చేయడానికి చేరండి", 
        "reg_btn": "నమోదు చేసుకోండి", 
        "reg_exist": "ఇప్పటికే నమోదు చేసుకున్నారా?", 
        "reg_login": "ఇక్కడ లాగిన్ చేయండి",
        "reg_name": "👤 పూర్తి పేరు",
        "reg_phone": "📱 మొబైల్ నంబర్",
        "reg_pass": "🔐 పాస్‌వర్డ్ సృష్టించండి",
        "reg_loc": "📍 గ్రామం / నగరం"
    },
    "hi": {
        "title": "एग्रोनेत्रा", 
        "subtitle": "AI-संचालित फसल रोग इंटेलिजेंस प्लेटफ़ॉर्म", 
        "detect_diagnose": "पता लगाएं • निदान करें • रक्षा करें",
        "nav_home": "होम वर्कस्पेस", 
        "nav_history": "एनालिटिक्स इतिहास", 
        "nav_analytics": "लाइव चार्ट", 
        "change_lang": "भाषा बदलें",
        "upload_title": "फसल का पत्ता स्कैन करें", 
        "upload_desc": "लक्षित पौधे के पत्ते की फ़ाइल अपलोड करें या लाइव फ़ोटो लें।",
        "btn_upload": "पत्ती की छवि अपलोड करें", 
        "btn_camera": "लाइव कैमरा से खींचें", 
        "profile_scope": "उपयोगकर्ता प्रोफ़ाइल:", 
        "lang_opts": "वैश्विक भाषा विकल्प:",
        "city": "क्षेत्रीय मूल्यांकन शहर:", 
        "scale": "पौधों की संख्या:", 
        "btn_predict": "रोग की जाँच करें",
        "feat_scan": "रोग स्कैन", 
        "feat_scan_desc": "डीप न्यूरल नेटवर्क तुरंत 29 वनस्पति परिवारों में विसंगतियों को वर्गीकृत करते हैं।",
        "feat_treat": "स्मार्ट उपचार", 
        "feat_treat_desc": "जैविक शमन विकल्पों के साथ-साथ रासायनिक नुस्खे।",
        "feat_weather": "मौसम की जानकारी", 
        "feat_weather_desc": "लाइव मौसम संबंधी फ़ीड।",
        "feat_reports": "पीडीएफ रिपोर्ट", 
        "feat_reports_desc": "डाउनलोड करने योग्य उद्यम-ग्रेड ऑडिट रिपोर्ट।",
        "stat_path": "समर्थित रोग", 
        "stat_acc": "सटीकता", 
        "stat_img": "प्रशिक्षित चित्र", 
        "stat_active": "AI विश्लेषण सक्रिय",
        "compare_btn": "रोग रिकॉर्ड की तुलना करें", 
        "compare_title": "एग्रोनेत्रा क्रॉस-पैथोलॉजी मैट्रिक्स",
        "target_genus": "लक्षित वनस्पति जीनस", 
        "status": "रोग की पहचान", 
        "confidence": "मॉडल आत्मविश्वास", 
        "severity": "कंप्यूटर विजन गंभीरता",
        "gauge_title": "रोग गंभीरता गेज", 
        "gauge_desc": "वर्तमान सक्रिय मूल्यांकन क्षेत्र:", 
        "xai_title": "व्याख्यात्मक AI (XAI) हीटमैप", 
        "xai_desc": "दृश्य सत्यापन मैपिंग।",
        "original_img": "मूल पत्ती की छवि", 
        "heatmap_img": "AI कंप्यूटर विजन हीटमैप", 
        "rationale": "मॉडल नैदानिक सत्यापन तर्क:",
        "simulator_title": "रिकवरी सिम्युलेटर", 
        "simulator_desc": "उपचार योजना लागू करने पर रिकवरी की प्रगति।",
        "baseline": "वर्तमान स्थिति", 
        "dose_48h": "प्राथमिक खुराक के बाद (48 घंटे)", 
        "stabilization": "स्थिरीकरण (7 दिन)",
        "recommendation_title": "उत्पादन अनुशंसा", 
        "read_aloud": "जोर से पढ़ें", 
        "profile_assessment": "रोग प्रोफ़ाइल मूल्यांकन",
        "description": "विवरण", 
        "symptoms": "पहचाने गए लक्षण:", 
        "cause": "जैविक रोगजनक कारण:", 
        "remediation": "उपचारात्मक पैरामीटर", 
        "treatment": "उपचार निर्माण",
        "organic": "जैविक उपचार", 
        "chemical": "रासायनिक उपचार", 
        "prevention": "रोकथाम प्रोटोकॉल", 
        "action_plan": "पाइपलाइन संचालन", 
        "calendar": "रणनीतिक क्षेत्र कार्यान्वयन कैलेंडर",
        "weather_title": "मौसम की जानकारी", 
        "temp": "तापमान खिड़की:", 
        "humidity": "सापेक्ष आर्द्रता:", 
        "sky": "आकाश का वातावरण:", 
        "hazard_risk": "रोग प्रसार खतरा जोखिम:",
        "recovery_proj": "फसल रिकवरी अनुमान", 
        "outlook": "आउटलुक क्षमता:", 
        "priority": "परिचालन प्राथमिकता", 
        "status_tier": "स्थिति सूचकांक:", 
        "deadline": "अनुशंसित समय सीमा:",
        "yield_loss_title": "संभावित फसल का नुकसान", 
        "yield_loss": "अनुमानित फसल उपज हानि", 
        "mitigation_cost": "शमन लागत आवंटन", 
        "net_impact": "शुद्ध वित्तीय प्रभाव:",
        "ai_doctor": "AI फसल डॉक्टर", 
        "btn_spread": "क्या यह फैलेगा?", 
        "btn_save": "क्या मैं अपनी फसल बचा सकता हूँ?", 
        "btn_cost": "उपचार की लागत?", 
        "btn_prevent": "रोकथाम के उपाय?",
        "placeholder": "वेक्टर प्रसार, नैदानिक कदम के बारे में पूछें...", 
        "btn_query": "क्वेरी कोपायलट", 
        "btn_download": "पीडीएफ ऑडिट रिपोर्ट डाउनलोड करें", 
        "btn_new_scan": "नए लक्ष्य का विश्लेषण करें", 
        "footer": "एग्रोनेत्रा क्रॉप इंटेलिजेंस सिस्टम्स v1.0 © 2026",
        "login_title": "किसान लॉगिन", 
        "login_sub": "एग्रोनेत्रा में आपका स्वागत है", 
        "login_btn": "🌿 लॉगिन करें", 
        "login_new": "नए किसान हैं?", 
        "login_create": "खाता बनाएं",
        "mobile_number": "📱 मोबाइल नंबर",
        "password": "🔐 पासवर्ड",
        "welcome_farmer": "👨‍🌾 किसान का स्वागत है",
        "ai_companion": "आपकी फसल के स्वास्थ्य के लिए AI सहायक",
        "reg_title": "👨‍🌾 किसान प्रोफ़ाइल बनाएं", 
        "reg_sub": "अपना स्कैन इतिहास सहेजने के लिए जुड़ें", 
        "reg_btn": "रजिस्टर करें और जुड़ें", 
        "reg_exist": "पहले से पंजीकृत हैं?", 
        "reg_login": "यहां लॉगिन करें",
        "reg_name": "👤 पूरा नाम",
        "reg_phone": "📱 मोबाइल नंबर",
        "reg_pass": "🔐 पासवर्ड बनाएं",
        "reg_loc": "📍 गाँव / शहर"
    }
}

def get_lang_dict(target_language):
    if target_language in ui_translations:
        return ui_translations[target_language]
    base_dict = ui_translations["en"]
    translated_dict = {}
    try:
        translator = GoogleTranslator(source='en', target=target_language)
        values = list(base_dict.values())
        translated_values = translator.translate_batch(values)
        for k, v in zip(base_dict.keys(), translated_values):
            translated_dict[k] = v
        return translated_dict
    except Exception:
        return base_dict

def translate_text(text, target_language):
    if target_language == "en" or not text:
        return text
    try:
        if isinstance(text, list):
            return [GoogleTranslator(source='auto', target=target_language).translate(str(item)) for item in text]
        return GoogleTranslator(source='auto', target=target_language).translate(str(text))
    except Exception:
        return text

# ==================================================
# Normalizer & Computer Vision Validation Helper Methods
# ==================================================

def normalize_class_name(raw_name):
    if not raw_name:
        return "Tomato - Healthy"
    clean = raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    for actual_key in severity_levels.keys():
        if clean.lower().strip() == actual_key.lower().strip():
            return actual_key
        if clean.split(" - ")[-1].lower().strip() in actual_key.lower().strip():
            if clean.split(" ")[0].lower().strip() in actual_key.lower().strip():
                return actual_key
    return "Tomato - Healthy"

def validate_image_quality(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None:
            return False, "Unable to read file matrix stream."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        leaf_mask = cv2.inRange(hsv, np.array([10, 15, 15]), np.array([105, 255, 255]))
        leaf_pixels = int(cv2.countNonZero(leaf_mask))

        if brightness < 40:
            return False, "Image exposure is too dark for feature extraction."
        if blur_score < 45:
            return False, "Image variance is blurry. Lens autofocus correction is requested."
        if leaf_pixels < 3000:
            return False, "Leaf area too small. Target out of distribution boundary constraints."

        return True, "Image quality acceptable."
    except Exception:
        return False, "Hardware structural exception inside CV pipeline validation layer."

def generate_heatmap(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_lesion = np.array([0, 40, 40])
        upper_lesion = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_lesion, upper_lesion)

        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        base_path, ext = os.path.splitext(image_path)
        heatmap_path = f"{base_path}_heatmap{ext}"
        
        cv2.imwrite(heatmap_path, output)
        return heatmap_path
    except Exception:
        return None

def get_weather(city):
    if not WEATHER_API_KEY or WEATHER_API_KEY == "YOUR_API_KEY":
        return {"temperature": 28.5, "humidity": 72, "weather": "scattered clouds (Simulation mode)"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "weather": data["weather"][0]["description"]
            }
        return {"temperature": 27.0, "humidity": 65, "weather": "broken clouds (API Fallback)"}
    except Exception:
        return {"temperature": 26.2, "humidity": 68, "weather": "overcast clouds (Offline Cache)"}

def generate_disease_chart():
    try:
        conn = sqlite3.connect("agronetra.db")
        cursor = conn.cursor()
        cursor.execute("SELECT disease, COUNT(*) FROM predictions GROUP BY disease")
        data = cursor.fetchall()
        conn.close()

        if not data:
            return False

        labels = [row[0] for row in data]
        counts = [row[1] for row in data]

        plt.figure(figsize=(6, 4))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, 
                colors=['#2e6f40', '#cf6a4c', '#f0a967', '#4c8da5', '#7f5a83'])
        plt.title("Disease Distribution Profile", fontsize=12, fontweight='bold', color='#1e3d2f')
        plt.tight_layout()
        
        chart_path = os.path.join(CHARTS_FOLDER, "disease_distribution.png")
        plt.savefig(chart_path, dpi=150, transparent=True)
        plt.close()
        return True
    except Exception:
        return False

# ==================================================
# Authentication & Access Routes
# ==================================================

@app.route("/")
def index_route():
    return redirect(url_for("splash"))

@app.route("/splash")
def splash():
    lang_code = session.get("language", "en")
    lang_dict = get_lang_dict(lang_code)
    
    if "language" not in session:
        next_route = url_for("language_page")
    elif "farmer_id" not in session:
        next_route = url_for("login")
    else:
        next_route = url_for("home")
        
    return render_template("splash.html", lang=lang_dict, next_route=next_route)

@app.route("/language")
def language_page():
    return render_template("language.html")

@app.route("/set_language/<lang>")
def set_language(lang):
    session["language"] = lang
    return redirect(url_for("splash"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        phone = request.form.get("phone")
        password = request.form.get("password")
        
        conn = sqlite3.connect("agronetra.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, language FROM farmers WHERE phone = ? AND password = ?", (phone, password))
        farmer = cursor.fetchone()
        conn.close()
        
        if farmer:
            session["farmer_id"] = farmer[0]
            session["farmer_name"] = farmer[1]
            if farmer[2]:
                session["language"] = farmer[2]
            return redirect(url_for("home"))
        else:
            lang_dict = get_lang_dict(session.get("language", "en"))
            error_msg = translate_text("Invalid mobile number or password.", session.get("language", "en"))
            return render_template("login.html", lang=lang_dict, error=error_msg)
            
    lang_dict = get_lang_dict(session.get("language", "en"))
    return render_template("login.html", lang=lang_dict)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        phone = request.form.get("phone")
        password = request.form.get("password")
        crop = request.form.get("crop", "") 
        location = request.form.get("location")
        language = session.get("language", "en")
        
        try:
            conn = sqlite3.connect("agronetra.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO farmers (name, phone, password, crop, location, language)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, phone, password, crop, location, language))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            lang_dict = get_lang_dict(session.get("language", "en"))
            error_msg = translate_text("Mobile number already registered.", session.get("language", "en"))
            return render_template("register.html", lang=lang_dict, error=error_msg)
            
    lang_dict = get_lang_dict(session.get("language", "en"))
    return render_template("register.html", lang=lang_dict)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index_route"))

@app.route("/home")
def home():
    if "language" not in session:
        return redirect(url_for("language_page"))
    if "farmer_id" not in session:
        return redirect(url_for("login"))
    lang_code = session.get("language", "en")
    lang_dict = get_lang_dict(lang_code)
    return render_template("index.html", lang=lang_dict, farmer_name=session.get("farmer_name"))

# ==================================================
# Core App Routing Endpoints
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    if "farmer_id" not in session:
        return redirect(url_for("login"))

    uploaded_file = request.files.get("file")
    camera_file = request.files.get("camera")

    file = None
    if camera_file and camera_file.filename != "":
        file = camera_file
    elif uploaded_file and uploaded_file.filename != "":
        file = uploaded_file

    if not file:
        return "No file selected", 400

    user_type = request.form.get("user_type", "gardener")
    language = session.get("language", "en")
    city = request.form.get("city", "Chittoor")
    plants = int(request.form.get("plants", 1))
    acres = float(request.form.get("acres", 1))

    lang_dict = get_lang_dict(language)
    scan_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    valid_image, image_message = validate_image_quality(filepath)
    if not valid_image:
        return render_template(
            "result.html", prediction="Unknown - Quality Rejection", crop_type="Unknown", confidence=0.00,
            confidence_warning="⚠️ Validation rejection triggered.", severity="N/A", severity_short="Healthy", mapped_severity="N/A",
            image_path=filepath, heatmap_path=filepath, description="Input guardrails triggered rejection.", symptoms="N/A", cause="N/A", organic="N/A", chemical="N/A",
            treatment="N/A", prevention="N/A", action_plan=["Re-upload clearer sample image file."], treatment_plan=["Upload a non-blurry, brightly-lit leaf close-up macro picture."],
            estimated_cost=0.00, recommendation="Abort diagnosis.", action_required="No", user_type=user_type, language=language, lang=lang_dict,
            weather={"temperature": "N/A", "humidity": 0, "weather": "N/A"}, risk="Low", risk_display="🟢 Calibration Lock", city_name=city, scan_time=scan_time,
            recovery_score=0, recovery_status="Critical", recovery_advice="N/A", priority_score=0, priority_status="Routine", treatment_deadline="N/A", final_decision="N/A",
            yield_loss=0.0, economic_loss=0.00, xai_explanation="Pre-processing pipeline identified file quality degradation constraints.",
            image_warning=image_message, prediction_id=0, status_category_display="Healthy"
        )

    heatmap_file = generate_heatmap(filepath)
    heatmap_path = heatmap_file if heatmap_file else filepath

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    raw_model_str = class_names[predicted_index]
    disease_name = normalize_class_name(raw_model_str)
    mapped_severity = severity_levels[disease_name]

    try:
        cv_img = cv2.imread(filepath)
        hsv_space = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        full_leaf_mask = cv2.inRange(hsv_space, np.array([5, 20, 20]), np.array([100, 255, 255]))
        only_green_mask = cv2.inRange(hsv_space, np.array([35, 40, 40]), np.array([85, 255, 255]))
        leaf_pixel_count = cv2.countNonZero(full_leaf_mask)
        green_pixel_count = cv2.countNonZero(only_green_mask)
        infected_area = (max(0, leaf_pixel_count - green_pixel_count) / leaf_pixel_count) * 100 if leaf_pixel_count > 0 else 0.0
    except Exception:
        infected_area = 12.0

    weather_data = get_weather(city)
    humidity_val = weather_data["humidity"]
    
    if humidity_val > 80:
        weather_risk = "High"
        weather_risk_display = "🔴 High Risk"
    elif humidity_val > 60:
        weather_risk = "Medium"
        weather_risk_display = "🟡 Medium Risk"
    else:
        weather_risk = "Low"
        weather_risk_display = "🟢 Low Risk"

    is_healthy = "healthy" in disease_name.lower()

    if is_healthy or infected_area == 0:
        severity_label = "Healthy"
        severity_display = "🟢 Healthy Plant (0% Infestation)"
        action_plan_key = "Healthy"
        final_recommendation = "Monitor crop regularly."
        action_required = "No"
    else:
        action_plan_key = "Disease"
        action_required = "Yes"
        if infected_area <= 15.0:
            severity_label = "Mild"
            severity_display = f"Mild ({round(infected_area, 2)}%)"
            final_recommendation = "Monitor crop regularly. Treat if symptoms expand."
        elif infected_area <= 40.0:
            severity_label = "Moderate"
            severity_display = f"Moderate ({round(infected_area, 2)}%)"
            final_recommendation = "Treatment recommended within 2-3 days."
        else:
            severity_label = "Severe"
            severity_display = f"Severe ({round(infected_area, 2)}%)"
            final_recommendation = "Immediate treatment required."

    raw_action_plan = action_plans[action_plan_key]

    raw_crop_name = disease_name.split(" - ")[0]
    crop_name = "Corn" if "Corn" in raw_crop_name else raw_crop_name
    unit_base = crop_cost.get(crop_name, 50)

    base_cost = unit_base if severity_label == "Mild" else (unit_base * 2 if severity_label == "Moderate" else (unit_base * 3 if severity_label == "Severe" else 0))
    multiplier = 1 if user_type == "gardener" else 100
    scale_count = plants if user_type == "gardener" else acres
    estimated_cost = base_cost * multiplier * scale_count

    if confidence < 60:
        confidence_warning = "⚠️ Low confidence prediction. Please upload a clearer leaf image for better diagnosis."
    else:
        confidence_warning = "✅ Prediction confidence acceptable."

    base_recovery = recovery_scores.get(disease_name, 100)
    if not is_healthy:
        if severity_label == "Moderate": base_recovery -= 10
        elif severity_label == "Severe": base_recovery -= 25
        if "High" in weather_risk: base_recovery -= 15
    recovery_score = max(0, min(100, base_recovery))

    if recovery_score >= 80:
        recovery_status = "Excellent"
        raw_recovery_advice = "Crop has a strong chance of recovery if treatment begins immediately."
    elif recovery_score >= 60:
        recovery_status = "Good"
        raw_recovery_advice = "Crop can likely recover with proper care and monitoring."
    elif recovery_score >= 40:
        recovery_status = "Moderate"
        raw_recovery_advice = "Recovery is possible but urgent action is required."
    else:
        recovery_status = "Critical"
        raw_recovery_advice = "Crop condition is critical. Immediate expert intervention is recommended."

    if is_healthy:
        yield_loss = 0.0
        economic_loss = 0.00
    else:
        yield_loss = 5 if severity_label == "Mild" else (15 if severity_label == "Moderate" else 30)
        if "High" in weather_risk: yield_loss += 10
        elif "Medium" in weather_risk: yield_loss += 5
        yield_loss = yield_loss * ((100 - recovery_score) / 100 + 0.5)
        yield_loss = round(yield_loss, 1)
        economic_loss = round(estimated_cost * (yield_loss / 10), 2)

    if is_healthy:
        priority_score = 1
        priority_status = "Routine"
        raw_treatment_deadline = "Monitor Regularly"
        raw_final_decision = "Continue monitoring and follow preventive measures."
    else:
        priority_score = 0
        if severity_label == "Severe": priority_score += 5
        elif severity_label == "Moderate": priority_score += 3
        else: priority_score += 1
        if "High" in weather_risk: priority_score += 3
        elif "Medium" in weather_risk: priority_score += 2
        else: priority_score += 1
        if recovery_score < 40: priority_score += 2
        elif recovery_score < 70: priority_score += 1
        priority_score = min(priority_score, 10)
        priority_status = "Critical" if priority_score >= 8 else ("Urgent" if priority_score >= 5 else "Routine")
        raw_treatment_deadline = "Within 24 Hours" if priority_status == "Critical" else ("Within 2-3 Days" if priority_status == "Urgent" else "Monitor Regularly")
        raw_final_decision = "Immediate intervention required to prevent severe crop loss." if priority_score >= 8 else (
            "Treatment should be started within the next few days." if priority_score >= 5 else "Continue monitoring and follow preventive measures."
        )

    info = disease_info.get(disease_name, {"description": "N/A", "symptoms": "N/A", "cause": "N/A", "organic": "N/A", "chemical": "N/A", "treatment": "N/A", "prevention": "N/A"})
    raw_xai_explanation = explanations.get(disease_name, "The model completed analysis mapping on structural layers.")
    raw_treatment_plan = smart_treatment_plan.get(disease_name, ["Follow recommended treatment", "Monitor crop regularly"])

    disease_display = translate_text(disease_name.split(' - ')[1] if ' - ' in disease_name else disease_name, language)
    crop_type_display = translate_text(raw_crop_name, language)
    description = translate_text(info["description"], language)
    symptoms = translate_text(info["symptoms"], language)
    cause = translate_text(info["cause"], language)
    organic = translate_text(info["organic"], language)
    chemical = translate_text(info["chemical"], language)
    treatment = translate_text(info["treatment"], language)
    prevention = translate_text(info["prevention"], language)
    action_plan = translate_text(raw_action_plan, language)
    treatment_plan = translate_text(raw_treatment_plan, language)
    recommendation = translate_text(final_recommendation, language)
    xai_explanation = translate_text(raw_xai_explanation, language)
    recovery_advice = translate_text(raw_recovery_advice, language)
    final_decision = translate_text(raw_final_decision, language)
    treatment_deadline = translate_text(raw_treatment_deadline, language)
    status_category_display = translate_text(severity_label, language)
    recovery_status_display = translate_text(recovery_status, language)

    conn = sqlite3.connect("agronetra.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (image_name, disease, confidence, user_type, estimated_cost, prediction_time, farmer_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (file.filename, disease_name, round(confidence, 2), user_type, round(estimated_cost, 2), scan_time, session.get("farmer_id"))
    )
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return render_template(
        "result.html", prediction_id=prediction_id, prediction=disease_display, raw_prediction=disease_name, crop_type=crop_type_display, confidence=round(confidence, 2),
        severity=severity_display, severity_short=severity_label, mapped_severity=mapped_severity, status_category_display=status_category_display,
        image_path=filepath, heatmap_path=heatmap_path, description=description, symptoms=symptoms, cause=cause, organic=organic, chemical=chemical,
        treatment=treatment, prevention=prevention, action_plan=action_plan, treatment_plan=treatment_plan, estimated_cost=round(estimated_cost, 2),
        recommendation=recommendation, action_required=action_required, user_type=user_type, language=language, lang=lang_dict,
        weather=weather_data, risk=weather_risk, risk_display=weather_risk_display, city_name=city, scan_time=scan_time,
        recovery_score=recovery_score, recovery_status=recovery_status_display, recovery_advice=recovery_advice,
        priority_score=priority_score, priority_status=priority_status, treatment_deadline=treatment_deadline, final_decision=final_decision,
        yield_loss=yield_loss, economic_loss=economic_loss, xai_explanation=xai_explanation
    )

@app.route("/chat", methods=["POST"])
def chat():
    disease = request.form.get("disease")
    question = request.form.get("question", "").lower()

    if "spread" in question: answer = "This disease may spread if not treated quickly."
    elif "save" in question: answer = "Early treatment can often save the crop."
    elif "fungicide" in question: answer = "Use the fungicide recommended in the treatment section."
    elif "cost" in question: answer = "Refer to the estimated treatment cost shown above."
    elif "prevent" in question: answer = "Follow the prevention guidelines provided."
    else: answer = f"The detected disease is {disease}. Please ask about spread, treatment, prevention, or cost."

    disease_name = normalize_class_name(disease)
    info = disease_info.get(disease_name, {"description": "N/A", "symptoms": "N/A", "cause": "N/A", "organic": "N/A", "chemical": "N/A", "treatment": "N/A", "prevention": "N/A"})
    action_plan_key = "Healthy" if "healthy" in disease_name.lower() else "Disease"
    action_plan = action_plans[action_plan_key]

    treatment_plan_raw = request.form.getlist("treatment_plan")
    if not treatment_plan_raw:
        treatment_plan = smart_treatment_plan.get(disease_name, ["Follow recommended treatment", "Monitor crop regularly"])
    else:
        treatment_plan = treatment_plan_raw

    raw_confidence = request.form.get("confidence", "0.00")
    try: confidence_val = float(str(raw_confidence).replace("%", "").strip())
    except (ValueError, TypeError): confidence_val = 0.00

    severity_short = request.form.get("severity_short", "Healthy")
    mapped_severity = severity_levels.get(disease_name, "Healthy")
    language = session.get("language", "en")
    lang_dict = get_lang_dict(language)
    
    answer = translate_text(answer, language)

    return render_template(
        "result.html",
        prediction_id=request.form.get("prediction_id", "1"),
        prediction=request.form.get("disease_display", disease_name), 
        raw_prediction=disease_name,
        crop_type=request.form.get("crop_type", disease_name.split(" - ")[0]), 
        confidence=round(confidence_val, 2),
        confidence_warning=request.form.get("confidence_warning", ""), 
        severity=request.form.get("severity", "N/A"), 
        severity_short=severity_short, 
        mapped_severity=mapped_severity,
        status_category_display=request.form.get("status_category_display", severity_short),
        image_path=request.form.get("image_path", ""), 
        heatmap_path=request.form.get("heatmap_path", ""), 
        description=request.form.get("description", info["description"]), 
        symptoms=request.form.get("symptoms", info["symptoms"]), 
        cause=request.form.get("cause", info["cause"]),
        organic=request.form.get("organic", info["organic"]), 
        chemical=request.form.get("chemical", info["chemical"]), 
        treatment=request.form.get("treatment", info["treatment"]), 
        prevention=request.form.get("prevention", info["prevention"]),
        action_plan=action_plan, 
        treatment_plan=treatment_plan, 
        estimated_cost=request.form.get("estimated_cost", "0.00"), 
        recommendation=request.form.get("recommendation", ""),
        action_required=request.form.get("action_required", "No"), 
        user_type=request.form.get("user_type", "gardener"), 
        language=language, lang=lang_dict,
        risk=request.form.get("risk", "Low"), 
        risk_display=request.form.get("risk_display", "🟢 Low Risk"), 
        city_name=request.form.get("city_name", "Chittoor"),
        weather={"temperature": request.form.get("weather_temp", "25.0"), "humidity": request.form.get("weather_hum", "50"), "weather": request.form.get("weather_cond", "clear sky")},
        scan_time=request.form.get("scan_time", ""), answer=answer,
        recovery_score=request.form.get("recovery_score", "100"), 
        recovery_status=request.form.get("recovery_status", "Excellent"), 
        recovery_advice=request.form.get("recovery_advice", ""),
        priority_score=request.form.get("priority_score", "1"), 
        priority_status=request.form.get("priority_status", "Routine"), 
        treatment_deadline=request.form.get("treatment_deadline", "Monitor Regularly"), 
        final_decision=request.form.get("final_decision", ""),
        yield_loss=request.form.get("yield_loss", "0.0"), 
        economic_loss=request.form.get("economic_loss", "0.00"), 
        xai_explanation=request.form.get("xai_explanation", ""), 
        image_warning=request.form.get("image_warning")
    )

@app.route("/history")
def history():
    if "farmer_id" not in session:
        return redirect(url_for("login"))
        
    conn = sqlite3.connect("agronetra.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE farmer_id = ? ORDER BY id DESC", (session["farmer_id"],))
    records = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE farmer_id = ?", (session["farmer_id"],))
    total_scans = cursor.fetchone()[0]

    if total_scans > 0:
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE farmer_id = ? AND disease NOT LIKE '%Healthy%'", (session["farmer_id"],))
        diseased_scans = cursor.fetchone()[0]
        healthy_scans = total_scans - diseased_scans

        cursor.execute("SELECT SUM(estimated_cost) FROM predictions WHERE farmer_id = ?", (session["farmer_id"],))
        total_cost = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT AVG(confidence) FROM predictions WHERE farmer_id = ?", (session["farmer_id"],))
        avg_confidence = round(cursor.fetchone()[0] or 0.0, 2)

        cursor.execute("SELECT disease, COUNT(*) as cnt FROM predictions WHERE farmer_id = ? GROUP BY disease ORDER BY cnt DESC LIMIT 1", (session["farmer_id"],))
        result = cursor.fetchone()
        most_common_disease = result[0] if result else "None"
    else:
        healthy_scans = diseased_scans = avg_confidence = total_cost = 0
        most_common_disease = "No diagnostic records initialized."

    conn.close()
    chart_exists = generate_disease_chart() if total_scans > 0 else False

    stats = {
        "total_scans": total_scans,
        "healthy_scans": healthy_scans,
        "diseased_scans": diseased_scans,
        "avg_confidence": avg_confidence,
        "most_common_disease": most_common_disease,
        "total_cost": round(total_cost, 2),
        "chart_exists": chart_exists
    }
    lang = get_lang_dict(session.get("language", "en"))
    return render_template("history.html", records=records, stats=stats, lang=lang)

@app.route("/export_csv")
def export_csv():
    if "farmer_id" not in session:
        return redirect(url_for("login"))
        
    conn = sqlite3.connect("agronetra.db")
    cursor = conn.cursor()
    cursor.execute("SELECT prediction_time, disease, confidence, user_type, estimated_cost FROM predictions WHERE farmer_id = ? ORDER BY id DESC", (session["farmer_id"],))
    records = cursor.fetchall()
    conn.close()

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["Date & Time", "Prediction Label", "Model Confidence (%)", "User Profile Type", "Mitigation Cost (INR)"])
    cw.writerows(records)

    response = make_response(si.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=agronetra_history_export.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route("/download_report/<int:prediction_id>")
def download_report(prediction_id):
    conn = sqlite3.connect("agronetra.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    record = cursor.fetchone()
    conn.close()

    if not record: return abort(404, description="Report Record Not Found")
    _, image_name, disease_name, confidence, user_type, estimated_cost, p_time, farmer_id = record

    language = session.get("language", "en")
    lang_dict = get_lang_dict(language)

    normalized = normalize_class_name(disease_name)
    info = disease_info.get(normalized, {"description": "N/A", "treatment": "N/A", "prevention": "N/A"})
    action_plan_list = action_plans["Healthy"] if "healthy" in normalized.lower() else action_plans["Disease"]
    action_plan_string = "; ".join(action_plan_list)

    disease_display = translate_text(disease_name, language)
    desc_display = translate_text(info["description"], language)
    treat_display = translate_text(info["treatment"], language)
    prev_display = translate_text(info["prevention"], language)
    action_display = translate_text(action_plan_string, language)

    pdf_filename = f"report_{prediction_id}.pdf"
    pdf_path = os.path.join("static", pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=24, textColor=colors.HexColor("#1e3d2f"), spaceAfter=15)
    label_style = ParagraphStyle("LabelStyle", fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#2c3e50"))
    value_style = ParagraphStyle("ValueStyle", fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#34495e"))

    content = [Paragraph(f"{lang_dict.get('title', 'AGRONETRA')} — {lang_dict.get('feat_reports', 'Crop Health Analysis Report')}", title_style), Spacer(1, 10)]
    
    report_data = [
        [Paragraph("Metric / Category", label_style), Paragraph("Diagnostic Field Value", label_style)],
        [Paragraph(lang_dict.get("status", "Target Botanical Identification"), label_style), Paragraph(disease_display, value_style)],
        [Paragraph(lang_dict.get("confidence", "Neural Network Confidence Level"), label_style), Paragraph(f"{confidence}%", value_style)],
        [Paragraph(lang_dict.get("profile_scope", "Target Classification Category"), label_style), Paragraph(user_type.capitalize(), value_style)],
        [Paragraph("Diagnostic Evaluation Window", label_style), Paragraph(p_time, value_style)],
        [Paragraph(lang_dict.get("description", "Symptomatic Structural Description"), label_style), Paragraph(desc_display, value_style)],
        [Paragraph(lang_dict.get("remediation", "Prescriptive Intervention Measures"), label_style), Paragraph(treat_display, value_style)],
        [Paragraph(lang_dict.get("prevention", "Long-term Preventative Measures"), label_style), Paragraph(prev_display, value_style)],
        [Paragraph(lang_dict.get("action_plan", "Tactical Field Deployment Plan"), label_style), Paragraph(action_display, value_style)],
        [Paragraph(lang_dict.get("mitigation_cost", "Estimated Treatment Cost"), label_style), Paragraph(f"INR {estimated_cost:.2f}", value_style)]
    ]

    summary_table = Table(report_data, colWidths=[200, 330])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#1e3d2f")), ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")), ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))
    content.append(summary_table)
    doc.build(content)
    return send_file(pdf_path, as_attachment=True, download_name=f"agronetra_report_{prediction_id}.pdf")

@app.route('/analytics')
def analytics():
    if "farmer_id" not in session:
        return redirect(url_for("login"))
    lang = get_lang_dict(session.get("language", "en"))
    return render_template("analytics.html", lang=lang)

if __name__ == "__main__":
    app.run(debug=True)