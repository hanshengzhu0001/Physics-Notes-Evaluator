import spacy
import pandas as pd
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.language import Language

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Read the existing dataset
df = pd.read_csv('statements_dataset.csv')

# Define custom NER component to extract physics-related entities
@Language.component("physics_entity_extractor")
def physics_entity_extractor(doc):
    matcher = Matcher(nlp.vocab)
    
    # Define patterns for physics-related entities
    patterns = [
        [{"LOWER": "force"}],
        [{"LOWER": "momentum"}],
        [{"LOWER": "mass"}],
        [{"LOWER": "velocity"}],
        [{"LOWER": "acceleration"}],
        [{"LOWER": "distance"}],
        [{"LOWER": "displacement"}],
        [{"LOWER": "speed"}],
        [{"LOWER": "time"}],
        [{"LOWER": "instantaneous"}],
        [{"LOWER": "work"}],
        [{"LOWER": "energy"}],
        [{"LOWER": "system"}],
        [{"LOWER": "object"}],
        [{"LOWER": "state"}],
        [{"LOWER": "rest"}],
        [{"LOWER": "motion"}],
        [{"LOWER": "net"}],
        [{"LOWER": "rate"}],
        [{"LOWER": "change"}],
        [{"LOWER": "body"}],
        [{"LOWER": "reaction"}],
        [{"LOWER": "thermal"}],
        [{"LOWER": "kinetic"}],
        [{"LOWER": "potential"}],
        [{"LOWER": "molecules"}],
        [{"LOWER": "temperature"}],
        [{"LOWER": "gas"}],
        [{"LOWER": "ideal"}],
        [{"LOWER": "intermolecular"}],
        [{"LOWER": "fusion"}],
        [{"LOWER": "solid"}],
        [{"LOWER": "capacity"}],
        [{"LOWER": "substance"}],
        [{"LOWER": "elementary"}],
        [{"LOWER": "entities"}],
        [{"LOWER": "atoms"}],
        [{"LOWER": "carbon"}],
        [{"LOWER": "constructive"}],
        [{"LOWER": "interference"}],
        [{"LOWER": "superposing"}],
        [{"LOWER": "waves"}],
        [{"LOWER": "phase"}],
        [{"LOWER": "difference"}],
        [{"LOWER": "path"}],
        [{"LOWER": "nλ"}],
        [{"LOWER": "destructive"}],
        [{"LOWER": "π"}],
        [{"LOWER": "incidence"}],
        [{"LOWER": "refraction"}],
        [{"LOWER": "monochromatic"}],
        [{"LOWER": "spreading"}],
        [{"LOWER": "wave"}],
        [{"LOWER": "obstacle"}],
        [{"LOWER": "normal"}],
        [{"LOWER": "line"}],
        [{"LOWER": "segment"}],
        [{"LOWER": "perpendicular"}],
        [{"LOWER": "surface"}],
        [{"LOWER": "arrival"}],
        [{"LOWER": "incident"}],
        [{"LOWER": "faraday’s law of induction"}],
        [{"LOWER": "electromotive force"}, {"LOWER": "emf"}],
        [{"LOWER": "electric field"}],
        [{"LOWER": "electric field strength"}, {"LOWER": "e"}],
        [{"LOWER": "electric resistance"}],
        [{"LOWER": "electric current"}],
        [{"LOWER": "electric potential"}],
        [{"LOWER": "half-wave rectification"}],
        [{"LOWER": "full-wave rectification"}],
        [{"LOWER": "ohm’s law"}],
        [{"LOWER": "lenz’s law"}],
        [{"LOWER": "magnetic flux"}],
        [{"LOWER": "time constant"}],
        [{"LOWER": "capacitance"}],
        [{"LOWER": "zero error"}],
        [{"LOWER": "left-hand rule"}],
        [{"LOWER": "resistivity"}],
        [{"LOWER": "resistor"}],
        [{"LOWER": "right-hand rule"}],
        [{"LOWER": "gravitational field strength"}],
        [{"LOWER": "newton’s universal law of gravitation"}],
        [{"LOWER": "escape velocity"}],
        [{"LOWER": "fundamental properties of matter"}],
        [{"LOWER": "mass defect"}],
        [{"LOWER": "exchange particle"}],
        [{"LOWER": "antiparticle"}],
        [{"LOWER": "pair annihilation"}],
        [{"LOWER": "pair production"}],
        [{"LOWER": "virtual particle"}],
        [{"LOWER": "radioactive decay"}],
        [{"LOWER": "radioisotope"}],
        [{"LOWER": "nuclear fusion"}],
        [{"LOWER": "nuclear fission"}],
        [{"LOWER": "meson"}],
        [{"LOWER": "baryon"}],
        [{"LOWER": "deep inelastic scattering"}],
        [{"LOWER": "isotope"}],
        [{"LOWER": "nuclide"}],
        [{"LOWER": "transmutation"}],
        [{"LOWER": "half-life"}],
        [{"LOWER": "electronvolt"}],
        [{"LOWER": "unified atomic mass unit"}],
        [{"LOWER": "wien’s law"}],
        [{"LOWER": "stefan-boltzmann's law"}],
        [{"LOWER": "black body"}],
        [{"LOWER": "emissivity"}],
        [{"LOWER": "albedo"}],
        [{"LOWER": "global warming"}],
        [{"LOWER": "hydroelectric power"}],
        [{"LOWER": "specific energy"}],
        [{"LOWER": "energy density"}],
        [{"LOWER": "primary energy"}],
        [{"LOWER": "secondary energy"}],
        [{"LOWER": "renewable resources"}],
        [{"LOWER": "non-renewable resources"}],
        [{"LOWER": "rayleigh criterion"}],
        [{"LOWER": "single-slit diffraction"}],
        [{"LOWER": "double-slit interference"}],
        [{"LOWER": "doppler effect"}],
        [{"LOWER": "total internal reflection"}],
        [{"LOWER": "standing waves"}],
        [{"LOWER": "maxima"}],
        [{"LOWER": "minima"}],
        [{"LOWER": "gravitational potential"}],
        [{"LOWER": "escape speed"}],
        [{"LOWER": "bohr model"}],
        [{"LOWER": "de broglie wavelength"}],
        [{"LOWER": "decay constant"}],
        [{"LOWER": "radioactive decay law"}],
        [{"LOWER": "wave function"}],
        [{"LOWER": "copenhagen interpretation"}],
        [{"LOWER": "cosmic scale factor"}],
        [{"LOWER": "parallax method"}],
        [{"LOWER": "main sequence stars"}],
        [{"LOWER": "planet"}],
        [{"LOWER": "comet"}],
        [{"LOWER": "asteroid"}],
        [{"LOWER": "red giant"}],
        [{"LOWER": "supernova"}],
        [{"LOWER": "neutron star"}],
        [{"LOWER": "black hole"}],
        [{"LOWER": "stellar cluster"}],
        [{"LOWER": "nebula"}],
        [{"LOWER": "cepheid variable"}],
        [{"LOWER": "luminosity"}],
        [{"LOWER": "apparent brightness"}],
        [{"LOWER": "hubble’s law"}],
        [{"LOWER": "big bang model"}],
        [{"LOWER": "cosmic microwave background radiation"}],
        [{"LOWER": "dark matter"}],
        [{"LOWER": "critical density"}],
        [{"LOWER": "dark energy"}],
        [{"LOWER": "type ia supernova"}],
        [{"LOWER": "stellar nucleosynthesis"}],
        [{"LOWER": "jeans criterion"}],
        [{"LOWER": "s-process"}],
        [{"LOWER": "r-process"}],
        [{"LOWER": "white dwarf"}],
        [{"LOWER": "chandrasekhar limit"}],
        [{"LOWER": "hubble constant"}],
        [{"LOWER": "redshift"}],
        [{"LOWER": "cosmic background radiation"}],

        # Unit symbols
        [{"LOWER": "newton"}],
        [{"LOWER": "n"}],
        [{"LOWER": "joule"}],
        [{"LOWER": "j"}],
        [{"LOWER": "watt"}],
        [{"LOWER": "w"}],
        [{"LOWER": "coulomb"}],
        [{"LOWER": "c"}],
        [{"LOWER": "volt"}],
        [{"LOWER": "v"}],
        [{"LOWER": "ohm"}],
        [{"LOWER": "ω"}],
        [{"LOWER": "tesla"}],
        [{"LOWER": "t"}],
        [{"LOWER": "weber"}],
        [{"LOWER": "wb"}],
        [{"LOWER": "farad"}],
        [{"LOWER": "f"}],
        [{"LOWER": "henry"}],
        [{"LOWER": "h"}],
        [{"LOWER": "siemens"}],
        [{"LOWER": "s"}],
        [{"LOWER": "becquerel"}],
        [{"LOWER": "bq"}],
        [{"LOWER": "gray"}],
        [{"LOWER": "gy"}],
        [{"LOWER": "sievert"}],
        [{"LOWER": "sv"}],
        [{"LOWER": "pascal"}],
        [{"LOWER": "pa"}],
        [{"LOWER": "lux"}],
        [{"LOWER": "lx"}],
        [{"LOWER": "lumen"}],
        [{"LOWER": "lm"}],
        [{"LOWER": "hertz"}],
        [{"LOWER": "hz"}],
        [{"LOWER": "kelvin"}],
        [{"LOWER": "k"}],
        [{"LOWER": "ampere"}],
        [{"LOWER": "a"}],
        [{"LOWER": "second"}],
        [{"LOWER": "s"}],
        [{"LOWER": "minute"}],
        [{"LOWER": "min"}],
        [{"LOWER": "hour"}],
        [{"LOWER": "h"}],
        [{"LOWER": "day"}],
        [{"LOWER": "d"}],
        [{"LOWER": "year"}],
        [{"LOWER": "yr"}],
        [{"LOWER": "n"}],
        [{"LOWER": "kg"}],
        [{"LOWER": "ms"}],
        [{"LOWER": "m"}],
        [{"LOWER": "v"}],
        [{"LOWER": "a"}],
        [{"LOWER": "g"}],
        [{"LOWER": "h"}],
        [{"LOWER": "KE"}],
        [{"LOWER": "PE"}],
        [{"LOWER": "mgh"}],
        [{"LOWER": "+"}],
        [{"LOWER": "-"}],
        [{"LOWER": "*"}],
        [{"LOWER": "/"}],
        #[{"IS_DIGIT": True}]  # Numbers
    ]
    
    matcher.add("PHYSICS_ENTITY", patterns)
    
    matches = matcher(doc)
    entities = []
    
    for match_id, start, end in matches:
        span = Span(doc, start, end, label="PHYSICS_ENTITY")
        entities.append((span.text, span.label_))
    
    doc._.physics_entities = entities
    return doc

# Register the custom extension attribute
Doc.set_extension("physics_entities", default=[], force=True)

# Add the custom component to the pipeline
nlp.add_pipe("physics_entity_extractor", last=True)

# Function to extract custom physics entities
def extract_physics_entities(statement):
    doc = nlp(statement)
    return doc._.physics_entities

# Apply the function to extract physics entities
df['physics_entities'] = df['statement'].apply(lambda x: extract_physics_entities(x))
#df['entities_str'] = df['physics_entities'].apply(lambda x: ' '.join([f"{ent[0]}({ent[1]})" for ent in x]))

# Display the DataFrame with entities
print(df)

# Save the modified DataFrame to a CSV file
df.to_csv('modified_physics_dataset.csv', index=False)
