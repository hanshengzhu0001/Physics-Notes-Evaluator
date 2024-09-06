import requests
import json

api_key = "d8b4c32aba2d498f9219b502ee4cfd0b"
statements = [
    "The shortest distance from the initial to the final position of an object in a specific direction is called displacement.",
    "The rate of change of velocity of an object is known as acceleration.",
    "An object in motion will remain in motion, and an object at rest will remain at rest unless acted upon by a net external force.",
    "Force is equal to the mass of an object multiplied by its acceleration (F = ma).",
    "For every action, there is an equal and opposite reaction.",
    "The energy an object possesses due to its motion is called kinetic energy, calculated as KE = 0.5 * m * v^2.",
    "The energy stored in an object due to its position or state is known as potential energy, such as gravitational potential energy (PE = mgh).",
    "The work done on an object is equal to the change in its kinetic energy.",
    "The change in momentum of an object when a force is applied over a time interval is calculated as Impulse = F * Δt.",
    "The total energy of an isolated system remains constant; energy can neither be created nor destroyed, only transformed from one form to another.",
    "The shortest distance from the initial to the final position of an object is known as displacement, regardless of direction.",
    "The rate of change of speed of an object is called acceleration.",
    "An object in motion will stop unless acted upon by a net external force.",
    "Force is equal to the mass of an object divided by its acceleration (F = m/a).",
    "For every action, there is an unequal and opposite reaction.",
    "The energy an object possesses due to its motion is called kinetic energy, calculated as KE = m * v^2.",
    "The energy stored in an object due to its position or state is known as kinetic energy (KE = mgh).",
    "The work done on an object is equal to the change in its potential energy.",
    "The change in velocity of an object when a force is applied over a time interval is calculated as Impulse = F * t.",
    "The total energy of an isolated system can be created and destroyed but only transformed from one form to another.",
    "The safety net increases the force by decreasing the time interval, as per the equation F = Δp/Δt."
]

# Define the endpoint (url), api-key (api-key is sent as an extra header)
api_endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
request_headers = {"x-api-key": api_key}

for statement in statements:
    # Define the payload (statement to be scored)
    payload = {"input_text": statement}

    # Send the POST request to the API and store the API response
    api_response = requests.post(url=api_endpoint, json=payload, headers=request_headers)

    # Print out the JSON payload the API sent back
    print(f"Statement: {statement}")
    print(f"API Response: {api_response.json()}")
    print()