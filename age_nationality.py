# Input from user
age = int(input("Enter your age: "))
nationality = input("Are you a citizen? (yes/no): ").lower()

# Check conditions with logical operators
if age > 18 and nationality == 'yes':
    print("You are eligible to cast your vote.")
elif age < 18 and nationality == 'yes':
    print("You are a minor and not eligible to vote.")
elif age < 18 and nationality == 'no':
    print("You are not eligible for nationality or voting.")
elif age > 18 and nationality == 'no':
    print("You are not eligible to cast your vote as you are not a citizen.")
