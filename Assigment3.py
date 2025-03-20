def can_administer_medicine(age, weight=None):
    # Check if the patient is an adult or meets the age and weight requirements
    if age >= 18:
        return "Medicine can be administered."
    elif 15 <= age < 18:
        if weight is not None and weight >= 55:
            return "Medicine can be administered."
        else:
            return "Medicine cannot be administered due to insufficient weight."
    else:
        return "Medicine cannot be administered to patients under 15 years old."

# Ask for the patient's age
age = int(input("Enter the patient's age: "))

# Check if the patient is an adolescent (15-17 years old) and ask for weight if so
if 15 <= age < 18:
    weight = float(input("Enter the patient's weight (in kg): "))
else:
    weight = None

# Call the function to determine if the medicine can be administered
result = can_administer_medicine(age, weight)
print(result)
