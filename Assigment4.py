def calculate_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'E'

# Ask the user for their score
score = float(input("Enter the student's score: "))

# Call the function to determine the grade
grade = calculate_grade(score)
print(f"The grade is: {grade}")
