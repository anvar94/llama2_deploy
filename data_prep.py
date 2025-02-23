import json

# Load JSON file
with open("/home/innail/바탕화면/anvar/YOLOv11-pt-master/GDP_country.json", "r") as file:
    data = json.load(file)[1]  # Extract list of country entries

# Convert to prompt-response format
formatted_data = []
for entry in data:
    country = entry.get("name", "Unknown")
    region = entry["region"]["value"].strip()
    income = entry["incomeLevel"]["value"]
    capital = entry["capitalCity"]
    longitude = entry.get("longitude", "N/A")
    latitude = entry.get("latitude", "N/A")

    # Skip records without meaningful data
    if country == "Unknown" or region == "Aggregates":
        continue

    # Create prompt-completion pairs
    prompt1 = f"What is the capital of {country}?"
    completion1 = f"The capital of {country} is {capital}."

    prompt2 = f"Which region does {country} belong to?"
    completion2 = f"{country} belongs to the {region} region."

    prompt3 = f"What is the income level of {country}?"
    completion3 = f"{country} has an income level of {income}."

    formatted_data.extend([
        {"prompt": prompt1, "completion": completion1},
        {"prompt": prompt2, "completion": completion2},
        {"prompt": prompt3, "completion": completion3},
    ])

# Save as JSON
with open("country_finetune.json", "w") as file:
    json.dump(formatted_data, file, indent=4)

print("✅ Data conversion completed!")
