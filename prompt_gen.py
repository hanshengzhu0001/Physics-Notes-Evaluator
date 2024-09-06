import openai
import csv
import time

# Set your OpenAI API key securely
api_key = ''
openai.api_key = api_key

# Known principles for validation
correct_principles = [
    "F = ma",
    "The force required to accelerate, in the direction of the force, a mass of 1 kg at 1 ms-2.",
    "Average speed is the speed over a period of time.",
    "The momentum in a closed system is constant.",
    "Instantaneous speed is the speed at a particular instant in time.",
    "Momentum (p) is calculated as p = mv, where m is mass and v is velocity.",
    "Work done is calculated as the force times the distance moved in the direction of the force.",
    "An object will continue in a state of rest or constant velocity (uniform motion) unless acted on by an external unbalanced force.",
    "The rate of change of momentum of a body is directly proportional to the force acting on the body.",
    "Every action has an equal and opposite reaction.",
    "The net force acting on the system is zero.",
    "Internal energy is the sum of kinetic energy and potential energy of molecules.",
    "Thermal energy is the energy transferred between two objects at different temperatures.",
    "An ideal gas is a gas in which there are no intermolecular forces, has zero PE and only KE.",
    "The specific latent heat of fusion is the energy absorbed when 1 kg of a solid melts at a constant temperature.",
    "The specific heat capacity is the energy required to change the temperature of a 1 kg substance by 1 K.",
    "A mole is the amount of substance that contains as many elementary entities as the number of atoms in 12 g of carbon-12.",
    "Constructive interference occurs when two superposing waves are in phase / phase difference is 0 / path difference is nλ.",
    "Destructive interference occurs when two superposing waves are out of phase / phase difference is π / path difference is (n + ½)λ.",
    "The angle of incidence where the angle of refraction is 90°.",
    "Monochromatic waves that are all in phase with each other (constant phase difference).",
    "Spreading out of a wave when it meets an obstacle.",
    "The normal is a line segment that is perpendicular to the surface at the arrival point of the incident ray.",
    "A cyclical (repeating) motion of an object from its central point. If the motion repeats in a fixed time interval it is called isochronous, and the time interval of repetition is called the period.",
    "Gravitational field strength is the gravitational force per unit mass at a point in a field.",
    "Newton’s Universal Law of Gravitation states that every point mass attracts every other point mass with a force proportional to the product of their masses and inversely proportional to the square of the distance between them.",
    "Escape velocity is the minimum speed needed for an object to escape the gravitational influence of a planet without further propulsion.",
    "The photoelectric effect is the emission of electrons from a metal surface when light of sufficient frequency shines on it.",
    "The Standard Model describes the electromagnetic, weak, and strong interactions of quarks and leptons.",
    "Binding energy is the energy required to split a nucleus into its constituent protons and neutrons.",
    "The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another.",
    "The first law of thermodynamics states that the change in internal energy of a system is equal to the heat added to the system minus the work done by the system.",
    "The second law of thermodynamics states that the total entropy of an isolated system can never decrease over time.",
    "The third law of thermodynamics states that as the temperature of a system approaches absolute zero, the entropy of the system approaches a minimum value.",
    "The uncertainty principle states that it is impossible to simultaneously know the exact position and exact momentum of a particle.",
    "Wave-particle duality is the concept that all particles exhibit both wave and particle properties.",
    "The principle of superposition states that when two or more waves overlap, the resulting wave is the sum of the individual waves.",
    "The Doppler effect is the change in frequency or wavelength of a wave relative to an observer moving relative to the wave source.",
    "The conservation of angular momentum states that the total angular momentum of a closed system remains constant if no external torques act on it.",
    "The principle of relativity states that the laws of physics are the same in all inertial frames of reference.",
    "The equivalence principle states that the effects of gravity are indistinguishable from the effects of acceleration in a small region of space-time.",
    "Hubble’s Law states that the velocity at which a galaxy recedes is directly proportional to its distance from the observer.",
    "The Big Bang model describes the universe's expansion from a hot, dense initial state."
]

def validate_statement(statement, correctness):
    if correctness == "incorrect":
        for principle in correct_principles:
            if principle.lower() in statement.lower():
                return False
    return True

def generate_statements(statement, correctness, n=25):
    if correctness == "correct":
        prompt = f"Generate {n} diverse correct statements related to the following sentences by varying sentence structures, non-physics word choices, and changing order of words:\n\n{statement}\n\nDiverse correct statements:"
    else:
        prompt = f"Generate {n} diverse incorrect statements related to the following sentences by varying sentence structures, word choices, and changing order of words and physics quantities\n\n{statement}\n\nDiverse incorrect statements:"
    
    response = None
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                n=1,
                max_tokens=1500,
                temperature=0.7
            )
            break  # Break if request is successful
        except openai.error.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise  # Re-raise exception if max retries exceeded

    statements = response.choices[0].message['content'].strip().split("\n")
    valid_statements = [s for s in statements if validate_statement(s, correctness)]
    return valid_statements

correct_statements = [
    # Topic 7: Atomic, Nuclear, and Particle Physics
    "Fundamental properties of matter include mass and charge, and all matter is composed of quarks and leptons.",
    "The photoelectric effect is the emission of electrons from a metal surface when light of sufficient frequency shines on it.",
    "Atomic energy levels are quantized, meaning electrons can only occupy certain energy levels.",
    "The work function is the minimum energy needed to remove an electron from a solid.",
    "Stationary states are specific energy levels where electrons do not radiate energy despite experiencing acceleration.",
    "The Standard Model describes the electromagnetic, weak, and strong interactions of quarks and leptons.",
    "Binding energy is the energy required to split a nucleus into its constituent protons and neutrons.",
    "Mass defect is the difference between the mass of a nucleus and the sum of the masses of its constituent nucleons.",
    "An elementary particle is a particle that is not composed of smaller particles.",
    "An exchange particle transmits one of the fundamental forces in particle interactions.",
    "An antiparticle has the same mass as its corresponding particle but opposite charge and quantum numbers.",
    "Pair annihilation occurs when a particle and its antiparticle collide and convert their mass into energy.",
    "Pair production occurs when a photon with sufficient energy creates a particle-antiparticle pair.",
    "A virtual particle is a particle that mediates the fundamental forces between other particles.",
    "Radioactive decay is the spontaneous transformation of an unstable atomic nucleus into a lighter nucleus.",
    "A radioisotope is an isotope that undergoes radioactive decay.",
    "Nuclear fusion is the process of combining two light atomic nuclei to form a heavier nucleus.",
    "Nuclear fission is the process of splitting a heavy atomic nucleus into lighter nuclei.",
    "A meson is a particle composed of a quark and an antiquark.",
    "A baryon is a particle composed of three quarks.",
    "Deep inelastic scattering involves high-energy electrons scattering off nucleons, revealing the internal structure of protons and neutrons.",
    "An isotope is an atom with the same number of protons but a different number of neutrons.",
    "A nuclide is a specific type of atomic nucleus with a certain number of protons and neutrons.",
    "Transmutation is the change of one element into another through nuclear reactions.",
    "Half-life is the time required for half of the radioactive nuclei in a sample to decay.",
    "An electronvolt is the energy gained by an electron when accelerated through a potential difference of one volt.",
    "The unified atomic mass unit is one twelfth of the mass of a carbon-12 atom.",
    
    # Topic 8: Energy Production
    "Wien’s Law states that the peak wavelength of radiation emitted by a black body is inversely proportional to its temperature.",
    "Stefan-Boltzmann's Law states that the total energy radiated per unit surface area of a black body is proportional to the fourth power of its temperature.",
    "A black body is an idealized physical body that absorbs all incident electromagnetic radiation.",
    "Emissivity is the ratio of the energy radiated by an object to the energy radiated by a black body at the same temperature.",
    "Albedo is the fraction of solar energy reflected by a surface.",
    "Global warming refers to the long-term increase in Earth's average surface temperature due to human activities.",
    "Hydroelectric power generates electricity by using the energy of flowing water.",
    "Nuclear fission releases energy by splitting heavy atomic nuclei.",
    "Nuclear fusion releases energy by combining light atomic nuclei.",
    "Nuclear power plants generate electricity by using nuclear reactions to produce heat, which is then used to generate steam that drives turbines.",
    "Specific energy is the energy per unit mass of a fuel.",
    "Energy density is the energy per unit volume of a fuel.",
    "Primary energy is energy found in nature that has not been subjected to any conversion or transformation process.",
    "Secondary energy is energy produced from primary energy sources.",
    "Renewable resources are energy sources that can be replenished naturally over short periods.",
    "Non-renewable resources are energy sources that are depleted faster than they are replenished.",
    
    # Topic 9: Wave Phenomena
    "The Rayleigh Criterion states that two images are just resolved when the central maximum of one diffraction pattern coincides with the first minimum of the other.",
    "Single-slit diffraction produces a pattern of light and dark fringes due to the spreading of light waves.",
    "Double-slit interference produces a pattern of bright and dark fringes due to the constructive and destructive interference of light waves.",
    "The Doppler effect is the change in frequency or wavelength of a wave relative to an observer moving relative to the wave source.",
    "A transverse wave has particle displacement perpendicular to the direction of wave propagation.",
    "A longitudinal wave has particle displacement parallel to the direction of wave propagation.",
    "Polarized light has electric field vectors oscillating in one plane.",
    "Total internal reflection occurs when a wave is completely reflected at the boundary between two media, with the angle of incidence exceeding the critical angle.",
    "Standing waves are formed by the interference of two waves traveling in opposite directions, creating nodes and antinodes.",
    "Maxima in standing waves are points of constructive interference, while minima are points of destructive interference.",
    
    # Topic 10: Fields
    "Electric potential is the work done per unit charge in moving a positive test charge from infinity to a point in an electric field.",
    "Gravitational potential is the work done per unit mass in moving a small test mass from infinity to a point in a gravitational field.",
    "Escape speed is the minimum speed required for an object to escape the gravitational influence of a planet without further propulsion.",
    
    # Topic 11: Quantum and Nuclear Physics
    "The Bohr model of the hydrogen atom describes electrons orbiting the nucleus in discrete energy levels.",
    "De Broglie wavelength is associated with a particle and is given by λ = h/p, where h is Planck’s constant and p is the particle's momentum.",
    "The decay constant is the probability per unit time that a given nucleus will decay.",
    "Half-life is the time required for half of the radioactive nuclei in a sample to decay.",
    "The photoelectric effect is the emission of electrons from a metal surface when illuminated by light of sufficient frequency.",
    "The radioactive decay law is N = N₀e^{-λt}, where N is the number of undecayed nuclei, N₀ is the initial number, and λ is the decay constant.",
    "The wave function describes the quantum state of a particle, with its square giving the probability density of finding the particle.",
    "The work function is the minimum energy needed to remove an electron from the surface of a metal.",
    "The unified atomic mass unit is defined as one twelfth of the mass of a carbon-12 atom.",
    "The Copenhagen Interpretation posits that a particle exists in all states simultaneously until it is observed.",
    
    # Topic 12: Astrophysics
    "The cosmic scale factor relates the relative size of the universe at different times, given by R/R₀ = 1 + z, where z is the redshift.",
    "The parallax method measures the position of a star at two different times and uses the shift in position to determine distance.",
    "Main sequence stars fuse hydrogen into helium in their cores.",
    "A planet is a celestial body that orbits a star, is spherical, and has cleared its orbit of other debris.",
    "A comet is an icy body that releases gas and dust, forming a glowing coma and tails when close to the Sun.",
    "An asteroid is a small rocky body orbiting the Sun.",
    "Medium-sized stars evolve into red giants, then shed their outer layers as planetary nebulae, leaving behind a white dwarf.",
    "Massive stars evolve into red supergiants, then explode as supernovae, leaving behind neutron stars or black holes.",
    "A stellar cluster is a group of stars that formed together and are gravitationally bound.",
    "A nebula is a cloud of gas and dust in space, often the birthplace of stars.",
    "Cepheid variables are stars whose luminosity changes periodically, used as standard candles to measure cosmic distances.",
    "Luminosity is the total amount of energy emitted by a star or other astronomical object per unit time.",
    "Apparent brightness is the observed brightness of a star from Earth, dependent on luminosity and distance.",
    "Hubble’s Law states that the velocity at which a galaxy recedes is directly proportional to its distance from the observer.",
    "The Big Bang model describes the universe's expansion from a hot, dense initial state.",
    "The cosmic microwave background radiation is the afterglow of the Big Bang, providing evidence for the universe's early state.",
    "Dark matter is a form of matter that does not emit or interact with electromagnetic radiation, inferred from gravitational effects on visible matter.",
    "The critical density is the density needed for the universe to have zero curvature and continue expanding forever.",
    "Dark energy is a mysterious form of energy causing the accelerated expansion of the universe.",
    "Type Ia supernovae are used as standard candles because they have a consistent peak luminosity.",
    "Stellar nucleosynthesis is the process by which elements are formed in the interiors of stars through nuclear fusion.",
    "The Jeans Criterion states that a gas cloud will collapse to form a star if its gravitational energy exceeds its thermal energy.",
    "The s-process and r-process are nucleosynthesis processes that create heavy elements through slow and rapid neutron capture, respectively.",
    "White dwarfs are supported by electron degeneracy pressure, with the Chandrasekhar limit being the maximum mass they can have before collapsing.",
    "The Hubble constant is the rate of expansion of the universe, used to estimate the age of the universe.",
    "Redshift is the increase in wavelength of light from distant galaxies, indicating the universe's expansion.",
    "The cosmic background radiation supports the Big Bang theory and provides evidence for dark matter and dark energy."
]

# Generate similar statements
all_statements = []
for statement in correct_statements:
    correct_similar = generate_statements(statement, "correct", n=50)
    incorrect_similar = generate_statements(statement, "incorrect", n=50)
    all_statements.append((statement, "correct"))
    for correct in correct_similar:
        all_statements.append((correct.strip(), "correct"))
    for incorrect in incorrect_similar:
        all_statements.append((incorrect.strip(), "incorrect"))

# Save to CSV
with open('statements_dataset2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["statement", "correctness"])
    writer.writerows(all_statements)

print("Dataset saved to statements_dataset.csv")
