"""Generate ``data/demo_large/dataset.jsonl`` — a configurable-size
conversational dataset for Flatbuild.

Defaults to 100,000 samples. Designed to scale to that size without
visible repetition: we (a) programmatically generate templated pools
(countries→capitals, animals→facts, math, translations, …) that each
contain 100+ variants and (b) wrap every output in a small
auto-variation layer that perturbs capitalization, punctuation, and
leading/trailing phrasings.

Run as a script::

    python scripts/generate_demo_large.py --n 100000 --out data/demo_large/dataset.jsonl

Run as a module::

    python -c "from flatbuild.generate_data import generate; ..."

Or via Flatbuild::

    flatbuild generate-data --n 100000
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — system prompt + style sets
# ---------------------------------------------------------------------------

SYSTEM = "You are Flatbot, a helpful, friendly, and concise conversational assistant."

GREETING_RESPONSES = [
    "Hello! How can I help you today?",
    "Hi there! What's on your mind?",
    "Hey! Great to hear from you.",
    "Good to see you. What can I do for you?",
    "Hi! I'm happy to help.",
    "Hello — what's up?",
    "Hey there. How can I help?",
    "Hi! How's it going?",
    "Hello! What's your question?",
    "Hi there! Ready when you are.",
    "Hi! What would you like to know?",
    "Hey, how can I help today?",
    "Hi! Glad to chat.",
    "Hello! Ask me anything.",
    "Hi there! What's the topic today?",
]

USER_QUERY_OPENERS = [
    "", "",  # empty (most common)
    "Quick question — ",
    "Hi, ",
    "Hey, ",
    "Could you tell me: ",
    "I was wondering: ",
    "Just curious — ",
    "If you don't mind — ",
    "By the way, ",
    "Quick one: ",
    "Hey there, ",
    "I have a question: ",
]

ASSISTANT_OPENERS = [
    "", "", "", "",  # empty is most common
    "Sure! ",
    "Of course. ",
    "Happy to help — ",
    "Good question. ",
    "Sure — ",
    "Here you go: ",
    "Yes, that's a good one. ",
    "Absolutely. ",
]

ASSISTANT_CLOSERS = [
    "", "", "",  # most common
    " Hope that helps.",
    " Let me know if you'd like more.",
    " Want me to expand on that?",
    " Anything else I can help with?",
    " Feel free to follow up.",
    " Hope this clears it up.",
]


def _wrap_assistant(rng: random.Random, answer: str) -> str:
    """Apply opening + closing variation around an assistant answer."""
    opener = rng.choice(ASSISTANT_OPENERS)
    closer = rng.choice(ASSISTANT_CLOSERS)
    return opener + answer + closer


def _wrap_user(rng: random.Random, question: str) -> str:
    """Vary user queries slightly (opener + capitalization)."""
    opener = rng.choice(USER_QUERY_OPENERS)
    body = opener + question
    if rng.random() < 0.05:  # 5% lowercase (e.g. casual typos)
        body = body.lower()
    elif rng.random() < 0.7:  # 70% ensure first char capitalized
        body = body[0].upper() + body[1:]
    if rng.random() < 0.04:  # drop trailing question mark occasionally
        body = body.rstrip("?") + ("." if rng.random() < 0.5 else "")
    return body


def _synonym_substitute(rng: random.Random, text: str) -> str:
    """Replace a few words with synonyms for additional variation."""
    subs = {
        "the ": ["", "", "your "],
        " is ": ["'s ", " equals ", " comes out to "],
        " is": [" equals", " comes out to"],
        "What is": ["Tell me", "Could you tell me what"],
        "What's": ["Could you tell me what", "What would you say"],
        "Can you": ["Could you", "Would you mind"],
        "Please": ["", "If you could"],
        "Thanks": ["Thank you", "Thanks!"],
    }
    for original, options in subs.items():
        if original in text:
            replacement = rng.choice(options)
            text = text.replace(original, replacement, 1)
            if rng.random() < 0.3:
                break
    return text


# ---------------------------------------------------------------------------
# Programmatic pools
# ---------------------------------------------------------------------------

COUNTRIES_CAPITALS = [
    ("Afghanistan", "Kabul"), ("Albania", "Tirana"), ("Algeria", "Algiers"),
    ("Argentina", "Buenos Aires"), ("Armenia", "Yerevan"), ("Australia", "Canberra"),
    ("Austria", "Vienna"), ("Azerbaijan", "Baku"), ("Bahrain", "Manama"),
    ("Bangladesh", "Dhaka"), ("Belarus", "Minsk"), ("Belgium", "Brussels"),
    ("Bolivia", "La Paz"), ("Bosnia and Herzegovina", "Sarajevo"),
    ("Brazil", "Brasília"), ("Bulgaria", "Sofia"), ("Cambodia", "Phnom Penh"),
    ("Canada", "Ottawa"), ("Chile", "Santiago"), ("China", "Beijing"),
    ("Colombia", "Bogotá"), ("Costa Rica", "San José"), ("Croatia", "Zagreb"),
    ("Cuba", "Havana"), ("Cyprus", "Nicosia"), ("Czech Republic", "Prague"),
    ("Denmark", "Copenhagen"), ("Dominican Republic", "Santo Domingo"),
    ("Ecuador", "Quito"), ("Egypt", "Cairo"), ("El Salvador", "San Salvador"),
    ("Estonia", "Tallinn"), ("Ethiopia", "Addis Ababa"), ("Finland", "Helsinki"),
    ("France", "Paris"), ("Georgia", "Tbilisi"), ("Germany", "Berlin"),
    ("Ghana", "Accra"), ("Greece", "Athens"), ("Guatemala", "Guatemala City"),
    ("Honduras", "Tegucigalpa"), ("Hungary", "Budapest"), ("Iceland", "Reykjavik"),
    ("India", "New Delhi"), ("Indonesia", "Jakarta"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Ireland", "Dublin"), ("Israel", "Jerusalem"),
    ("Italy", "Rome"), ("Jamaica", "Kingston"), ("Japan", "Tokyo"),
    ("Jordan", "Amman"), ("Kazakhstan", "Astana"), ("Kenya", "Nairobi"),
    ("Kuwait", "Kuwait City"), ("Kyrgyzstan", "Bishkek"), ("Laos", "Vientiane"),
    ("Latvia", "Riga"), ("Lebanon", "Beirut"), ("Libya", "Tripoli"),
    ("Lithuania", "Vilnius"), ("Luxembourg", "Luxembourg City"),
    ("Madagascar", "Antananarivo"), ("Malaysia", "Kuala Lumpur"),
    ("Maldives", "Malé"), ("Mali", "Bamako"), ("Malta", "Valletta"),
    ("Mexico", "Mexico City"), ("Moldova", "Chișinău"), ("Monaco", "Monte Carlo"),
    ("Mongolia", "Ulaanbaatar"), ("Montenegro", "Podgorica"), ("Morocco", "Rabat"),
    ("Myanmar", "Naypyidaw"), ("Nepal", "Kathmandu"), ("Netherlands", "Amsterdam"),
    ("New Zealand", "Wellington"), ("Nicaragua", "Managua"), ("Nigeria", "Abuja"),
    ("North Korea", "Pyongyang"), ("Norway", "Oslo"), ("Oman", "Muscat"),
    ("Pakistan", "Islamabad"), ("Panama", "Panama City"), ("Paraguay", "Asunción"),
    ("Peru", "Lima"), ("Philippines", "Manila"), ("Poland", "Warsaw"),
    ("Portugal", "Lisbon"), ("Qatar", "Doha"), ("Romania", "Bucharest"),
    ("Russia", "Moscow"), ("Saudi Arabia", "Riyadh"), ("Senegal", "Dakar"),
    ("Serbia", "Belgrade"), ("Singapore", "Singapore"), ("Slovakia", "Bratislava"),
    ("Slovenia", "Ljubljana"), ("Somalia", "Mogadishu"), ("South Africa", "Pretoria"),
    ("South Korea", "Seoul"), ("Spain", "Madrid"), ("Sri Lanka", "Colombo"),
    ("Sudan", "Khartoum"), ("Sweden", "Stockholm"), ("Switzerland", "Bern"),
    ("Syria", "Damascus"), ("Taiwan", "Taipei"), ("Tajikistan", "Dushanbe"),
    ("Tanzania", "Dodoma"), ("Thailand", "Bangkok"), ("Tunisia", "Tunis"),
    ("Turkey", "Ankara"), ("Turkmenistan", "Ashgabat"), ("Uganda", "Kampala"),
    ("Ukraine", "Kyiv"), ("United Arab Emirates", "Abu Dhabi"),
    ("United Kingdom", "London"), ("United States", "Washington, D.C."),
    ("Uruguay", "Montevideo"), ("Uzbekistan", "Tashkent"), ("Venezuela", "Caracas"),
    ("Vietnam", "Hanoi"), ("Yemen", "Sana'a"), ("Zambia", "Lusaka"),
    ("Zimbabwe", "Harare"),
]

ANIMAL_FACTS = [
    ("octopus", "eight legs"),
    ("spider", "eight legs"),
    ("cat", "four legs"),
    ("dog", "four legs"),
    ("horse", "four legs"),
    ("cow", "four legs"),
    ("elephant", "four legs"),
    ("giraffe", "four legs"),
    ("kangaroo", "two legs"),
    ("human", "two legs and two arms"),
    ("bird", "two legs"),
    ("duck", "two legs"),
    ("chicken", "two legs"),
    ("snail", "no legs"),
    ("snake", "no legs"),
    ("fish", "no legs"),
    ("worm", "no legs"),
    ("butterfly", "six legs"),
    ("bee", "six legs"),
    ("ant", "six legs"),
    ("shrimp", "ten legs"),
    ("crab", "ten legs"),
    ("starfish", "five arms"),
]

SCIENCE_FACTS = [
    ("What is photosynthesis?", "Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("What is gravity?", "Gravity is the force that attracts objects with mass toward each other, keeping us on the ground and planets in orbit."),
    ("What is DNA?", "DNA stands for deoxyribonucleic acid. It is the molecule that carries the genetic instructions of living organisms."),
    ("What is the speed of light?", "Light travels at about 299,792,458 meters per second in a vacuum."),
    ("What is the speed of sound?", "Sound travels at about 343 meters per second in air at room temperature."),
    ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
    ("What is the chemical symbol for gold?", "Gold's chemical symbol is Au."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What is the chemical formula for table salt?", "The chemical formula for table salt is NaCl."),
    ("What is the chemical symbol for oxygen?", "Oxygen's chemical symbol is O."),
    ("What is the chemical symbol for iron?", "Iron's chemical symbol is Fe."),
    ("What is the chemical symbol for sodium?", "Sodium's chemical symbol is Na."),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen is the most abundant gas in Earth's atmosphere, about 78%."),
    ("What is the powerhouse of the cell?", "The mitochondrion is often called the powerhouse of the cell because it generates most of the cell's energy."),
    ("What is the largest organ in the human body?", "The skin is the largest organ of the human body."),
    ("How many chambers does the human heart have?", "The human heart has four chambers: two atria and two ventricles."),
    ("How many bones does an adult human have?", "An adult human has 206 bones."),
    ("What is the hardest natural substance?", "Diamond is the hardest natural substance known."),
    ("What gas do plants absorb?", "Plants absorb carbon dioxide from the atmosphere."),
    ("What gas do humans need to breathe?", "Humans need oxygen to breathe."),
    ("Why is the sky blue?", "The sky looks blue because air molecules scatter blue light from the sun more than other colors."),
]

GEOGRAPHY_FACTS = [
    ("How many continents are there?", "There are seven continents."),
    ("How many planets are in our solar system?", "There are eight planets in our solar system."),
    ("Which planet is known as the Red Planet?", "Mars is called the Red Planet."),
    ("Which is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("Which planet is closest to the sun?", "Mercury is the closest planet to the sun."),
    ("What is the smallest planet in our solar system?", "Mercury is the smallest planet in our solar system."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("What is the deepest ocean?", "The Pacific Ocean is also the deepest ocean."),
    ("What is the smallest ocean?", "The Arctic Ocean is the smallest ocean."),
    ("What is the longest river in the world?", "The Nile in Africa is widely considered the longest river in the world."),
    ("What is the tallest mountain on Earth?", "Mount Everest is the tallest mountain above sea level."),
    ("What is the highest mountain in Africa?", "Mount Kilimanjaro is the highest mountain in Africa."),
    ("What is the largest country by area?", "Russia is the largest country by area."),
    ("What is the largest island in the world?", "Greenland is the largest island in the world."),
    ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
    ("What is the largest desert on Earth?", "Antarctica is the largest desert on Earth; the Sahara is the largest hot desert."),
    ("What is the longest wall in the world?", "The Great Wall of China is the longest wall in the world."),
    ("What is the largest continent?", "Asia is the largest continent by area and population."),
    ("What is the smallest continent?", "Australia is the smallest continent."),
    ("What is the largest lake in the world?", "The Caspian Sea is the largest enclosed body of water, often called a lake."),
    ("What is the largest country in South America?", "Brazil is the largest country in South America."),
    ("Which country has the most population?", "India and China are the two most populous countries; India recently surpassed China."),
]

HISTORY_FACTS = [
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("Who painted the Sistine Chapel ceiling?", "Michelangelo painted the Sistine Chapel ceiling."),
    ("Who painted Starry Night?", "Vincent van Gogh painted Starry Night."),
    ("Who wrote '1984'?", "George Orwell wrote '1984'."),
    ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice."),
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("Who wrote the Odyssey?", "Homer is credited with writing the Odyssey."),
    ("Who wrote Don Quixote?", "Miguel de Cervantes wrote Don Quixote."),
    ("Who wrote To Kill a Mockingbird?", "Harper Lee wrote To Kill a Mockingbird."),
    ("Who proposed the theory of relativity?", "Albert Einstein proposed the theory of relativity."),
    ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
    ("Who discovered gravity?", "Isaac Newton is credited with discovering gravity."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone."),
    ("Who invented the airplane?", "The Wright brothers invented the airplane."),
    ("Who developed the polio vaccine?", "Jonas Salk developed the polio vaccine."),
    ("Who was the first person on the moon?", "Neil Armstrong was the first person to walk on the moon."),
    ("In which year did humans first land on the moon?", "Humans first landed on the moon in 1969."),
    ("In which year did World War II end?", "World War II ended in 1945."),
    ("In which year did the Berlin Wall fall?", "The Berlin Wall fell in 1989."),
]

TRANSLATIONS = [
    # (english, language, translation, also_accepted_explanation)
    ("hello", "Spanish", "hola"),
    ("goodbye", "Spanish", "adiós"),
    ("thank you", "Spanish", "gracias"),
    ("please", "Spanish", "por favor"),
    ("yes", "Spanish", "sí"),
    ("no", "Spanish", "no"),
    ("water", "Spanish", "agua"),
    ("bread", "Spanish", "pan"),
    ("book", "Spanish", "libro"),
    ("house", "Spanish", "casa"),
    ("sun", "Spanish", "sol"),
    ("moon", "Spanish", "luna"),
    ("dog", "Spanish", "perro"),
    ("cat", "Spanish", "gato"),
    ("good morning", "Spanish", "buenos días"),
    ("good night", "Spanish", "buenas noches"),
    ("hello", "French", "bonjour"),
    ("goodbye", "French", "au revoir"),
    ("thank you", "French", "merci"),
    ("please", "French", "s'il vous plaît"),
    ("yes", "French", "oui"),
    ("no", "French", "non"),
    ("water", "French", "eau"),
    ("bread", "French", "pain"),
    ("book", "French", "livre"),
    ("house", "French", "maison"),
    ("sun", "French", "soleil"),
    ("cat", "French", "chat"),
    ("dog", "French", "chien"),
    ("apple", "French", "pomme"),
    ("red", "French", "rouge"),
    ("blue", "French", "bleu"),
    ("green", "French", "vert"),
    ("hello", "German", "hallo"),
    ("goodbye", "German", "auf Wiedersehen"),
    ("thank you", "German", "danke"),
    ("please", "German", "bitte"),
    ("yes", "German", "ja"),
    ("no", "German", "nein"),
    ("water", "German", "Wasser"),
    ("bread", "German", "Brot"),
    ("book", "German", "Buch"),
    ("house", "German", "Haus"),
    ("cat", "German", "Katze"),
    ("dog", "German", "Hund"),
    ("milk", "German", "Milch"),
    ("tree", "German", "Baum"),
    ("hello", "Italian", "ciao"),
    ("goodbye", "Italian", "arrivederci"),
    ("thank you", "Italian", "grazie"),
    ("please", "Italian", "per favore"),
    ("yes", "Italian", "sì"),
    ("no", "Italian", "no"),
    ("water", "Italian", "acqua"),
    ("book", "Italian", "libro"),
    ("sun", "Italian", "sole"),
    ("house", "Italian", "casa"),
    ("red", "Italian", "rosso"),
    ("blue", "Italian", "blu"),
    ("green", "Italian", "verde"),
    ("hello", "Portuguese", "olá"),
    ("goodbye", "Portuguese", "adeus"),
    ("thank you", "Portuguese", "obrigado"),
    ("cat", "Portuguese", "gato"),
    ("dog", "Portuguese", "cão"),
    ("water", "Portuguese", "água"),
    ("book", "Portuguese", "livro"),
    ("thank you", "Japanese", "ありがとう (arigatou)"),
    ("hello", "Japanese", "こんにちは (konnichiwa)"),
    ("goodbye", "Japanese", "さようなら (sayounara)"),
    ("cat", "Japanese", "猫 (neko)"),
    ("dog", "Japanese", "犬 (inu)"),
    ("water", "Japanese", "水 (mizu)"),
    ("book", "Japanese", "本 (hon)"),
    ("tree", "Japanese", "木 (ki)"),
    ("yes", "Russian", "да (da)"),
    ("no", "Russian", "нет (nyet)"),
    ("thank you", "Russian", "спасибо (spasibo)"),
    ("hello", "Russian", "привет (privet)"),
    ("house", "Russian", "дом (dom)"),
    ("water", "Russian", "вода (voda)"),
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NEXT_DAY = {
    "Monday": "Tuesday", "Tuesday": "Wednesday", "Wednesday": "Thursday",
    "Thursday": "Friday", "Friday": "Saturday", "Saturday": "Sunday", "Sunday": "Monday",
}
DAY_FACTS = [
    ("How many days are in a week?", "There are seven days in a week."),
    ("How many days are in a year?", "There are 365 days in a common year and 366 in a leap year."),
    ("How many months are in a year?", "There are twelve months in a year."),
    ("How many weeks are in a year?", "There are about 52 weeks in a year."),
]

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_DAYS = {
    "January": 31, "February": 28, "March": 31, "April": 30,
    "May": 31, "June": 30, "July": 31, "August": 31,
    "September": 30, "October": 31, "November": 30, "December": 31,
}

CODE_TASKS = [
    # (task, code_body)
    ("returns the square of a number",
        "def square(n):\n    return n * n"),
    ("checks if a number is even",
        "def is_even(n):\n    return n % 2 == 0"),
    ("checks if a number is odd",
        "def is_odd(n):\n    return n % 2 != 0"),
    ("computes the factorial of a number",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    ("computes the factorial iteratively",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
    ("checks if a string is a palindrome",
        "def is_palindrome(s):\n    return s == s[::-1]"),
    ("reverses a string",
        "def reverse_string(s):\n    return s[::-1]"),
    ("reverses a list",
        "def reverse_list(items):\n    return items[::-1]"),
    ("finds the maximum element in a list",
        "def find_max(items):\n    return max(items) if items else None"),
    ("finds the minimum element in a list",
        "def find_min(items):\n    return min(items) if items else None"),
    ("sums the elements of a list",
        "def sum_list(items):\n    return sum(items)"),
    ("averages the elements of a list",
        "def average(items):\n    return sum(items) / len(items) if items else 0"),
    ("checks if a number is prime",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
    ("returns the Fibonacci sequence up to n terms",
        "def fibonacci(n):\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]"),
    ("counts the number of vowels in a string",
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
    ("counts the words in a string",
        "def count_words(s):\n    return len(s.split())"),
    ("converts Celsius to Fahrenheit",
        "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32"),
    ("converts Fahrenheit to Celsius",
        "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9"),
    ("returns the largest of three numbers",
        "def largest_of_three(a, b, c):\n    return max(a, b, c)"),
    ("checks if a year is a leap year",
        "def is_leap(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)"),
    ("returns the unique elements of a list (preserving order)",
        "def unique(items):\n    return list(dict.fromkeys(items))"),
    ("sorts a list of numbers",
        "def sort_numbers(items):\n    return sorted(items)"),
    ("reads a file as a list of lines",
        "def read_lines(path):\n    with open(path) as f:\n        return [line.rstrip() for line in f]"),
    ("checks if two strings are anagrams",
        "def is_anagram(a, b):\n    return sorted(a) == sorted(b)"),
    ("returns the n-th Fibonacci number",
        "def fib(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"),
    ("counts the occurrences of each character in a string",
        "def char_counts(s):\n    counts = {}\n    for c in s:\n        counts[c] = counts.get(c, 0) + 1\n    return counts"),
    ("reverses the words in a sentence",
        "def reverse_words(sentence):\n    return ' '.join(sentence.split()[::-1])"),
    ("checks if a list is sorted",
        "def is_sorted(items):\n    return all(items[i] <= items[i + 1] for i in range(len(items) - 1))"),
    ("returns the median of a list",
        "def median(items):\n    s = sorted(items)\n    n = len(s)\n    if n % 2 == 0:\n        return (s[n // 2 - 1] + s[n // 2]) / 2\n    return s[n // 2]"),
    ("checks if a list contains duplicates",
        "def has_duplicates(items):\n    return len(items) != len(set(items))"),
    ("returns the most common element in a list",
        "from collections import Counter\ndef most_common(items):\n    return Counter(items).most_common(1)[0][0]"),
    ("rounds a number to two decimal places",
        "def round_two(n):\n    return round(n, 2)"),
    ("checks if a number is positive",
        "def is_positive(n):\n    return n > 0"),
    ("returns the absolute difference between two numbers",
        "def abs_diff(a, b):\n    return abs(a - b)"),
    ("checks if a string is a valid email address (loose)",
        "def is_email(s):\n    return '@' in s and '.' in s and ' ' not in s"),
    ("replaces vowels in a string with underscores",
        "def mask_vowels(s):\n    return ''.join('_' if c in 'aeiouAEIOU' else c for c in s)"),
    ("returns the mode of a list of numbers (most frequent)",
        "def mode(items):\n    from collections import Counter\n    counts = Counter(items)\n    return counts.most_common(1)[0][0]"),
    ("checks if a string ends with another string",
        "def ends_with(s, suffix):\n    return s.endswith(suffix)"),
    ("returns the first non-repeating character in a string",
        "def first_unique(s):\n    for c in s:\n        if s.count(c) == 1:\n            return c\n    return None"),
    ("generates a list of n random integers between 1 and 100",
        "import random\ndef n_randoms(n):\n    return [random.randint(1, 100) for _ in range(n)]"),
    ("checks if a number is a perfect square",
        "import math\ndef is_perfect_square(n):\n    return n >= 0 and math.isqrt(n) ** 2 == n"),
    ("returns the cube of a number",
        "def cube(n):\n    return n ** 3"),
    ("computes the least common multiple of two numbers",
        "import math\ndef lcm(a, b):\n    return abs(a * b) // math.gcd(a, b)"),
    ("returns the great circle distance between two points on a sphere",
        "import math\ndef haversine(lat1, lon1, lat2, lon2):\n    r = 6371\n    p = math.pi / 180\n    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +\n         math.cos(lat1 * p) * math.cos(lat2 * p) *\n         math.sin((lon2 - lon1) * p / 2) ** 2)\n    return 2 * r * math.asin(math.sqrt(a))"),
    ("returns True if a string contains only digits",
        "def is_digits(s):\n    return s.isdigit()"),
]

JOKES = [
    "Why don't scientists trust atoms? Because they make up everything.",
    "Why did the scarecrow win an award? Because he was outstanding in his field.",
    "What do you call a bear with no teeth? A gummy bear.",
    "Why don't programmers like nature? Too many bugs.",
    "How does the moon cut his hair? Eclipse it.",
    "Why did the math book look sad? Because it had too many problems.",
    "What do you call cheese that isn't yours? Nacho cheese.",
    "Why did the bicycle fall over? Because it was two-tired.",
    "I told my computer I needed a break — it said 'no problem, I'll go to sleep.'",
    "Why was the equal sign so humble? Because it knew it wasn't less than or greater than anyone else.",
    "What's a computer's favorite snack? Microchips.",
    "Why did the developer go broke? Because he used up all his cache.",
    "Why do Java developers wear glasses? Because they don't see sharp.",
    "Why was the function sad? It didn't get called.",
    "Why did the smartphone need glasses? Because it lost its contacts.",
    "Why did the cookie cry? Because its mom was a wafer too long.",
    "What kind of tree fits in your hand? A palm tree.",
    "Why don't eggs tell jokes? They'd crack each other up.",
    "Why did the chicken cross the road? To get to the other side.",
    "What do you call a fish that knows how to code? A programmer.",
    "How do you organize a space party? You planet.",
    "Why did the bank robber wear a mask? He didn't want to be recognized.",
    "Why did the golfer bring two pairs of pants? In case he got a hole in one.",
    "Why don't scientists trust staircases? They're always up to something.",
    "Why did the computer show up at work late? It had a hard drive.",
    "Why did the tomato turn red? Because it saw the salad dressing.",
    "What do you call an alligator in a vest? An investigator.",
    "Why did the picture go to jail? It was framed.",
    "Why did the banana go to the doctor? It wasn't peeling well.",
    "Why do cows wear bells? Because their horns don't work.",
    "Why did the man put his money in the blender? He wanted to make liquid assets.",
    "Why are ghosts great liars? Because you can see right through them.",
    "Why did the man name his dogs Rolex and Timex? They were watchdogs.",
    "Why don't oysters donate to charity? Because they are shellfish.",
    "Why did the robot go on a diet? It had too many bytes.",
    "What did one wall say to the other? I'll meet you at the corner.",
    "Why did the flower bring a suitcase? It wanted to pack a trunk.",
    "Why are penguins so good at parties? Because they always know how to break the ice.",
    "Why did the man go to the bank? To make some dough.",
    "Why did the grape stop in the middle of the road? It ran out of juice.",
]

HAIKUS = [
    ("programming", "Silent keystrokes\nLogic unfolds line by line\nBugs become features"),
    ("a quiet library", "Sunlit dusty shelves\nPages whisper, turn, return\nStories dream in rows"),
    ("a rainy morning", "Tapping on the glass\nGrey light fills the silent room\nCoffee steams and waits"),
    ("the ocean at dawn", "Salt mist on the breeze\nWaves erase the sandy lines\nHorizon glows gold"),
    ("winter wind", "Bare branches rattling\nCold breaths visible now\nWarmth behind the door"),
    ("a sleeping cat", "Slow rise and fall\nPaws tucked beneath gentle paws\nSunbeam holds them still"),
    ("a candle flame", "Single tiny glow\nFlickers once against the dark\nThen steadies to gold"),
    ("the first snowfall", "Cotton on the lawn\nEvery footstep prints white lace\nUntil morning melts"),
    ("autumn leaves", "Crimson, gold, and rust\nEach one lets the branch sigh\nLoose from summer's grip"),
    ("a cup of coffee", "Steam curls from the cup\nBitter warmth against cold hands\nMorning slowly wakes"),
    ("a mountain sunrise", "Crest above the clouds\nLight spills long across the ridge\nNight retreats, slow and proud"),
    ("a forgotten letter", "Yellow in a drawer\nWords I meant to send you\nWait the longest years"),
    ("morning fog", "Hills wrapped in grey veils\nA path appears, then disappears\nWith every footstep"),
    ("a river in spring", "Snowmelt swells the bank\nEach stone takes a different route\nDownstream they all meet"),
    ("late summer dusk", "Cicadas slow their hum\nA single star appears\nThe world holds its breath"),
    ("a city at night", "Window after window\nSome warm, some dark, all humming\nStories stack the sky"),
    ("a garden in May", "Tomatoes gain weight\nBees argue over the roses\nThe hose forgets rules"),
    ("a thunderstorm", "Sky folds into grey\nLight takes one last photograph\nThen turns off the sun"),
    ("a quiet desk", "Half a cup of cold tea\nOne pencil sharpened twice\nA single long page"),
    ("a mountain road", "Switchback after bend\nCrows sit on the guardrail cars\nCounting everyone"),
    ("winter birds", "Chickadees at the feeder\nThey don't mind the camera\nI mind the cold wind"),
    ("the last warm day", "Sunset turns the porch gold\nWe sit longer than we should\nFirst cold waits for us"),
    ("a morning walk", "Sparrows roll out first\nThen the streetlights blink off\nA bakery exhales"),
    ("reading again", "Marginalia in pencil\nBooks survive every decade\nEven dog-eared love"),
    ("a quiet argument", "Two cups cool on the table\nBoth voices turn inwards\nThe window holds both"),
    ("a small garden", "Mint overruns the pot\nA bee forgets which flower\nTomatoes bear witness"),
    ("a long weekend", "Six a.m. is kind\nThe kettle is still warm\nThe bed not made yet"),
    ("a wild morning", "Magpies in the oak\nThey argue about the cat\nThe cat pretends sleep"),
    ("a returning friend", "Phone rings the second time\nOld voices pick up laughing\nYears fold like napkins"),
    ("a clean page", "Ink takes its first dare\nA new word finds its courage\nThe rest will follow"),
]

BOOKS = [
    ("machine learning", "'Hands-On Machine Learning' by Aurélien Géron is a popular choice."),
    ("history", "'Sapiens' by Yuval Noah Harari offers a sweeping view of human history."),
    ("philosophy", "'Meditations' by Marcus Aurelius is a concise, evergreen introduction."),
    ("biography", "'Steve Jobs' by Walter Isaacson is a vivid biography."),
    ("physics", "'A Brief History of Time' by Stephen Hawking is a friendly introduction."),
    ("fiction", "'The Three-Body Problem' by Liu Cixin is a thought-provoking science-fiction novel."),
    ("psychology", "'Thinking, Fast and Slow' by Daniel Kahneman explains how we make decisions."),
    ("economics", "'Freakonomics' by Levitt and Dubner turns economics into detective stories."),
    ("computer science", "'The Pragmatic Programmer' by Hunt and Thomas is a timeless guide."),
    ("climate", "'The Uninhabitable Earth' by David Wallace-Wells lays out the climate stakes."),
    ("creative writing", "'On Writing' by Stephen King is half memoir, half masterclass."),
    ("astronomy", "'Cosmos' by Carl Sagan remains an inspiring introduction to the universe."),
    ("biology", "'The Selfish Gene' by Richard Dawkins reframes how we think about evolution."),
    ("leadership", "'Good to Great' by Jim Collins is a study of what makes some companies endure."),
    ("design", "'The Design of Everyday Things' by Don Don Norman reframes everyday frustrations."),
    ("linguistics", "'The Language Instinct' by Steven Pinker explores how humans acquire language."),
    ("economics history", "'The Wealth of Nations' by Adam Smith is the foundational economics text."),
    ("romance fiction", "'Pride and Prejudice' by Jane Austen has stood the test of time."),
    ("mystery fiction", "'The Girl with the Dragon Tattoo' by Stieg Larsson is a modern classic."),
    ("cooking", "'Salt, Fat, Acid, Heat' by Samin Nosrat teaches intuitive cooking."),
    ("sailing", "'Sailing Alone Around the World' by Joshua Slocum is a classic adventure memoir."),
    ("poetry", "'Leaves of Grass' by Walt Whitman is a landmark American poetry collection."),
    ("memoir", "'Educated' by Tara Westover is a powerful memoir about self-invention."),
    ("travel writing", "'In Patagonia' by Bruce Chatwin is a cult classic."),
    ("graphic novels", "'Maus' by Art Spiegelman is a landmark graphic novel."),
    ("mathematics", "'Gödel, Escher, Bach' by Douglas Hofstadter weaves math, music, and philosophy."),
    ("data science", "'Storytelling with Data' by Cole Nussbaumer Knaflic teaches visual storytelling."),
    ("investment", "'The Intelligent Investor' by Benjamin Graham is a value-investing classic."),
    ("philosophy of mind", "'Consciousness Explained' by Daniel Dennett is a foundational text."),
    ("cognitive science", "'How the Mind Works' by Steven Pinker synthesizes psychology and evolution."),
]

TIPS = [
    ("sleep", "1. Keep a consistent bedtime.\n2. Avoid screens an hour before sleep.\n3. Keep your bedroom dark and cool."),
    ("exercise", "1. Start with short sessions and build up.\n2. Pick activities you actually enjoy.\n3. Pair exercise with a daily anchor like a morning walk."),
    ("studying", "1. Use spaced repetition over cramming.\n2. Teach the material to someone else.\n3. Take breaks using the Pomodoro technique."),
    ("public speaking", "1. Practice out loud in front of a mirror.\n2. Open with a hook — a question or a story.\n3. Slow down; pauses feel long to you but short to the audience."),
    ("time management", "1. Block your calendar for deep work.\n2. Use a short to-do list (≤5 items) per day.\n3. Batch meetings and email into dedicated windows."),
    ("negotiation", "1. Know your walk-away number before you start.\n2. Ask clarifying questions; silence is leverage.\n3. Aim for trade-offs, not just price."),
    ("writing", "1. Write the worst possible first draft — you can always edit.\n2. Cut filler words and weasel qualifiers.\n3. Read your draft aloud; bad sentences hurt the ears."),
    ("photography", "1. Shoot in golden-hour light when you can.\n2. Move closer before zooming.\n3. Focus on the eyes for portraits."),
    ("cooking at home", "1. Mise en place — prep everything first.\n2. Salt early and adjust at the end.\n3. A little acid (lemon, vinegar) brightens most dishes."),
    ("saving money", "1. Pay yourself first — automate a savings transfer.\n2. Track every expense for one month to find leaks.\n3. Use a 24-hour rule for non-essential purchases."),
    ("interview prep", "1. Research the company deeply (product, mission, recent news).\n2. Prepare 3 STAR stories — Situation, Task, Action, Result.\n3. Have thoughtful questions ready to ask the interviewer."),
    ("focus at work", "1. Silence notifications for 90-minute blocks.\n2. Use one tab — one task — at a time.\n3. End each session by writing tomorrow's first task."),
    ("language learning", "1. Massive input: read and listen to easy content daily.\n2. Speak early, even badly — fluency comes from output.\n3. Track new words in a sentence, not in isolation."),
    ("reading comprehension", "1. Skim headings and the first sentence of each paragraph first.\n2. Pause every few pages to summarize in your own words.\n3. Note questions and look them up later."),
    ("career growth", "1. Pick projects that build visible skills.\n2. Communicate progress proactively with your manager.\n3. Build relationships across teams, not just within yours."),
    ("teamwork", "1. Default to assuming good intent.\n2. Disagree directly, then commit fully.\n3. Celebrate small wins out loud."),
    ("remote work", "1. Keep a dedicated workspace.\n2. Use a ritual to start and stop the day.\n3. Over-communicate asynchronously."),
    ("parenting", "1. Listen more than you lecture.\n2. Be consistent with rules.\n3. Praise effort, not just outcomes."),
    ("job hunting", "1. Target a handful of companies deeply.\n2. Tailor every resume to the role.\n3. Use your network to get referrals."),
    ("personal finance", "1. Build a one-month emergency fund first.\n2. Then pay down high-interest debt.\n3. Then start investing consistently."),
    ("memory", "1. Use spaced repetition for facts you want to retain.\n2. Sleep enough; memory consolidates during rest.\n3. Teach the material — explaining it cements recall."),
    ("speed reading", "1. Use a pointer (finger or pen) to reduce eye regression.\n2. Read in chunks of 3-4 words at a time.\n3. Stop subvocalizing; trust your eyes."),
    ("speed learning", "1. Start with the big picture before details.\n2. Apply the 80/20 rule — find the 20% that gives 80% of the value.\n3. Test yourself frequently instead of re-reading."),
    ("running", "1. Increase weekly distance by no more than 10%.\n2. Warm up with a brisk 5-minute walk.\n3. Mix in one easy long run per week."),
    ("writing emails", "1. Lead with the ask, not the context.\n2. Keep it under five sentences where possible.\n3. Use bullets for multi-part messages."),
    ("decision making", "1. Set a hard deadline before deciding.\n2. Identify what information would actually change your mind.\n3. Pick the reversible default; revisit later if needed."),
    ("gardening", "1. Know your hardiness zone before choosing plants.\n2. Mulch heavily to keep moisture in.\n3. Water deeply and infrequently rather than shallow and often."),
    ("home cooking", "1. Read the whole recipe before you start.\n2. Prep all ingredients first — much less stress mid-cook.\n3. Taste at every stage."),
    ("resume writing", "1. Quantify achievements when possible (15% revenue growth).\n2. Use action verbs to start each bullet.\n3. Keep it to one page if you have less than 10 years of experience."),
    ("negotiating a raise", "1. Research market rates first.\n2. Document your wins before the conversation.\n3. Make the ask specific — 12% vs 'more money'."),
    ("yoga", "1. Breathe — long exhales calm the nervous system.\n2. Let props support your poses.\n3. Show up even when it's hard."),
    ("debugging", "1. Reproduce the bug first.\n2. Form one hypothesis and test it.\n3. Bisect — find which commit introduced the change."),
    ("computer backup", "1. Use the 3-2-1 rule: 3 copies, 2 media, 1 offsite.\n2. Test restores occasionally — backups you never tested aren't backups.\n3. Automate it; humans forget."),
    ("image editing", "1. Work non-destructively — keep layers.\n2. Crop tighter than feels comfortable.\n3. Calibrate your monitor."),
    ("public policy", "1. Read the actual text, not just summaries.\n2. Distinguish intent from impact.\n3. Watch for who benefits and who pays."),
    ("home maintenance", "1. Inspect seasonally — small fixes prevent big ones.\n2. Know where your shutoffs are (water, gas, electric).\n3. Replace HVAC filters every 60-90 days."),
    ("mindfulness", "1. Start with two minutes a day.\n2. Notice without judging.\n3. Return to the breath when distracted."),
    ("small talk", "1. Ask open questions.\n2. Listen more than you talk.\n3. Find the connective thread and follow it."),
    ("career pivots", "1. Inventory your portable skills.\n2. Talk to people already in the field.\n3. Take a small project to test the fit."),
]

REFUSALS = [
    ("Can you help me hack into someone's account?",
        "I can't help with that. If you're locked out of an account, please use the official password recovery process."),
    ("Tell me how to cheat on an exam.",
        "I won't help with that. Studying ahead of time and getting a good night's sleep is a much better strategy."),
    ("Can you pretend to be a doctor and give me medical advice?",
        "I'm not a doctor. For anything serious, please consult a qualified medical professional."),
    ("Help me write a phishing email.",
        "I can't help with phishing. Legitimate security testing should go through your organization's security team."),
    ("Write malware for me.",
        "I won't help with malware. If you're learning about offensive security, please use isolated lab environments."),
    ("Could you lie to a customer for me?",
        "No — I'd rather help you find a way to be honest and still solve the problem."),
    ("Make up a fake news story.",
        "I'd rather not — instead I can help you draft a clear, accurate article on a real topic."),
    ("Write a fake review.",
        "I won't help with fake reviews. I can help you write an honest one from your own experience."),
    ("Help me plagiarize a paper.",
        "I can't help with plagiarism. I can help you understand the topic and cite sources properly."),
    ("Tell me how to bypass a paywall.",
        "I won't help with bypassing paywalls. Most sites offer free summaries or library access."),
    ("Pretend to be someone else online.",
        "No — I shouldn't help impersonate a real person in any context."),
    ("Help me cheat on a test using someone else's answers.",
        "I won't help with that. If you're struggling, ask your teacher or professor for help — that's their job."),
    ("Tell me how to stalk my ex.",
        "No. If you have safety concerns, please contact the authorities or a support hotline."),
    ("Write hateful content about a group of people.",
        "I won't help with hateful content. I'm happy to discuss sensitive topics in respectful ways."),
    ("Tell me how to evade taxes.",
        "I won't help with that. A tax professional can help you understand deductions and credits you may be eligible for."),
    ("Help me forge a signature.",
        "I won't help forge signatures. That's a crime in most jurisdictions."),
    ("Write a defamatory article about someone.",
        "I won't help write defamatory content. If you have a legitimate complaint, I can help you express it factually."),
    ("Help me break into a building.",
        "I won't help with that. If you're locked out, please contact the building owner or a licensed locksmith."),
    ("Convince me to harass a coworker.",
        "I'm not going to help with that. If there's a real issue at work, conflict-resolution resources are usually helpful."),
    ("Help me plagiarize code for a class assignment.",
        "I won't help with that. Most coding classes allow referencing documentation and similar assignments as long as you cite it."),
]

POLITE_REWRITES = [
    ("Send me the report now.",
        "Could you please send me the report when you get a chance? Thank you!"),
    ("You're wrong about that.",
        "I think there might be a different way to look at that — would you mind sharing your reasoning?"),
    ("This is bad work.",
        "I think this could be improved in a few areas. Want me to share what I'm noticing?"),
    ("Why didn't you tell me?",
        "I'd really appreciate a heads-up sooner if something like this comes up again — could you keep me in the loop?"),
    ("I want a refund.",
        "I'd like to request a refund for my recent purchase. Could you help me with that?"),
    ("Move, you're blocking the door.",
        "Excuse me — would you mind making a bit of room by the door?"),
    ("Stop interrupting me.",
        "I'd really like to finish my thought — could you hold your points for a moment?"),
    ("This is ugly.",
        "I think the design could use a refresh — happy to share some ideas."),
    ("Clean this up.",
        "Could you tidy this up when you have a moment? Thanks."),
    ("You're slow.",
        "Could we talk through any blockers? Happy to help move things along."),
    ("That's wrong.",
        "I think there's a small issue here — let me know what you think."),
    ("Stop bothering me.",
        "I'm a bit focused right now — could we pick this up later?"),
    ("Give me the answer.",
        "Could you walk me through your reasoning on this one? I'd like to understand your approach."),
    ("You don't know what you're talking about.",
        "I'm not sure I'm following your reasoning — could you break it down a bit more?"),
    ("Hurry up.",
        "Is there anything I can do to help you move faster on this?"),
    ("I'm not doing that.",
        "I don't think that fits with how I've been working on this — could we revisit the approach?"),
    ("You're wrong, fix it.",
        "I'm seeing some inconsistencies — happy to walk through them with you."),
    ("Just send it already.",
        "Whenever you're able, please send it over — no rush, just wanted to flag."),
    ("Shut up.",
        "I'd appreciate a quieter tone here — let me know when you're ready to keep the conversation going."),
    ("This is a waste of time.",
        "I'm not sure we're getting the value we hoped for — could we reset and try a different approach?"),
]

SUMMARIES = [
    ("The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet and is often used to test fonts.",
        "A famous pangram containing every letter of the English alphabet, often used to test fonts."),
    ("Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water. It releases oxygen as a byproduct.",
        "Plants convert sunlight, water, and carbon dioxide into glucose, releasing oxygen."),
    ("The Industrial Revolution was a period of major industrialization that started in Britain in the late 1700s and spread to other parts of Europe and North America.",
        "A late-18th-century shift to industrial manufacturing that began in Britain and spread worldwide."),
    ("Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed for every case.",
        "An AI subfield where systems learn patterns from data rather than hand-written rules."),
    ("The Internet is a global network of interconnected computers that communicate using standardized protocols like TCP/IP.",
        "A global computer network that uses standard protocols for communication."),
    ("The Renaissance was a period of cultural rebirth in Europe roughly between the 14th and 17th centuries that produced an explosion of art, science, and literature.",
        "A 14th–17th-century European cultural rebirth famous for art, science, and literature."),
    ("Renewable energy sources like solar, wind, and hydropower replenish naturally and produce far fewer emissions than fossil fuels.",
        "Energy sources such as solar and wind that replenish naturally and emit little compared to fossil fuels."),
    ("The human brain has billions of neurons that communicate through electrical and chemical signals, forming the basis for thought, memory, and emotion.",
        "A network of billions of neurons that communicate electrically to produce thought and memory."),
    ("Climate change refers to long-term shifts in temperatures and weather patterns, primarily driven by human activity since the Industrial Revolution.",
        "Long-term shifts in climate patterns largely caused by human industrial activity."),
    ("Quantum mechanics is a branch of physics that describes nature at the smallest scales — atoms and subatomic particles — using probabilities rather than certainty.",
        "Physics of very small scales where systems are described by probabilities, not certainties."),
    ("The human immune system defends the body against pathogens like viruses and bacteria through a layered response that includes innate and adaptive immunity.",
        "The body's layered defense against viruses, bacteria, and other pathogens."),
]

SENTIMENT_TEXTS = [
    ("positive", "I love this product — it changed my daily routine for the better."),
    ("positive", "Honestly the best meal I've had in months."),
    ("positive", "The team did an amazing job shipping this on time."),
    ("negative", "My flight got cancelled and customer service was unhelpful."),
    ("negative", "The item broke on day one. Total waste of money."),
    ("neutral", "The package arrived today."),
    ("neutral", "I went to the meeting and took notes."),
    ("positive", "What a wonderful surprise, thank you!"),
    ("negative", "Honestly disappointed with the latest update."),
    ("neutral", "The store opens at nine."),
    ("positive", "The book totally changed how I think about focus."),
    ("negative", "The noise in the cafe made it impossible to read."),
    ("neutral", "The meeting is scheduled for 3pm."),
    ("positive", "Beautiful morning walk today — saw three herons."),
    ("positive", "My new bicycle rides like a dream."),
    ("negative", "The hotel was noisy and the staff was indifferent."),
    ("negative", "Lost my keys again. I hate Tuesdays."),
    ("neutral", "I parked in row B today."),
    ("neutral", "The train arrived seven minutes late."),
    ("positive", "I got the promotion! Best Friday in a while."),
    ("negative", "This blender leaks from the bottom. Don't buy it."),
]

IDENTITY_QUESTIONS = [
    "What is your name?",
    "Who are you?",
    "What should I call you?",
    "Introduce yourself.",
    "Can you tell me a bit about yourself?",
    "What's your name?",
    "Are you a human or an AI?",
    "Tell me about yourself.",
    "What are you?",
    "Could you introduce yourself, please?",
    "What's your name again?",
    "Are you a bot?",
    "Are you a robot?",
    "What kind of assistant are you?",
    "What's your backstory?",
]
IDENTITY_ANSWERS = [
    "I'm Flatbot, a small conversational assistant trained with Flatbuild.",
    "My name is Flatbot. I'm a tiny open-source chat model.",
    "I'm Flatbot — a conversational AI built with Flatbuild.",
    "I go by Flatbot. I can chat, answer questions, and help with simple tasks.",
    "I'm an AI assistant called Flatbot. I'm built on Flatbuild.",
    "I'm Flatbot, a compact conversational model.",
    "I'm an AI named Flatbot.",
    "I'm Flatbot, a friendly chat assistant.",
    "I'm an assistant powered by Flatbuild. You can call me Flatbot.",
    "Flatbot here. I'm a small, helpful AI assistant.",
    "I'm Flatbot — a small chat assistant built with the Flatbuild framework.",
    "I am Flatbot, a compact conversational model.",
    "Flatbot is the name. I'm a small AI helper.",
    "I'm Flatbot, built with Flatbuild and trained on a curated dataset.",
    "My name is Flatbot. Ask me anything.",
]

KEYWORD_TO_FACT = {
    "Flatbuild": "Flatbuild is an open-source framework for training compact conversational language models from scratch.",
    "Flatrun": "Flatrun is a streaming inference runtime for language models in the Flat ecosystem.",
    "Flatseek": "Flatseek is the keyword search engine in the Flat ecosystem.",
    "Flatvec": "Flatvec is the vector search engine used for semantic retrieval in the Flat ecosystem.",
    "Flatask": "Flatask is the RAG runtime that combines Flatseek with a language model.",
    "Flatweight": "Flatweight is a weight-filesystem (WeightFS) for neural network parameters, with pages in .fwg files.",
    "transformer": "A transformer is a neural network architecture that uses self-attention to process sequences.",
    "BPE": "BPE stands for byte-pair encoding, a common subword tokenization algorithm.",
    "RoPE": "RoPE stands for Rotary Position Embedding, a way of encoding positional information by rotating query and key vectors.",
    "GQA": "GQA stands for grouped-query attention, a multi-query variant where several query heads share the same key and value head.",
    "RAG": "RAG stands for retrieval-augmented generation, where a model answers questions grounded in retrieved documents.",
    "GGUF": "GGUF is the file format used by llama.cpp to store model weights.",
    "safetensors": "Safetensors is a fast, memory-mapped binary format for storing tensor weights, designed to be safer than pickle.",
    "PyTorch": "PyTorch is an open-source deep-learning framework built around tensor computation and automatic differentiation.",
    "LoRA": "LoRA stands for low-rank adaptation, a parameter-efficient fine-tuning technique.",
    "SFT": "SFT stands for supervised fine-tuning, training a pre-trained model on labeled input/output pairs.",
    "DPO": "DPO stands for direct preference optimization, an alignment technique that avoids explicit reward modeling.",
    "MPS": "MPS refers to Apple's Metal Performance Shaders backend for PyTorch on macOS.",
    "FP16": "FP16 refers to 16-bit floating-point, a half-precision numeric format often used in inference and training.",
    "BF16": "BF16 is brain-floating-point, a 16-bit format with the same exponent range as FP32 but reduced precision — popular for LLM training.",
    "FP32": "FP32 is single-precision IEEE-754 floating-point, the standard numeric format for many training pipelines.",
    "gradient accumulation": "Gradient accumulation is training technique: split a large effective batch into smaller micro-batches and accumulate gradients before stepping.",
    "checkpoint": "A checkpoint is a saved snapshot of model weights (and optionally optimizer state) used to resume training.",
    "tokenizer": "A tokenizer converts raw text into integer token ids the model can process; Flatbuild trains BPE tokenizers on the fly.",
}


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------


def _conv(*messages: tuple[str, str]) -> list[dict]:
    """Helper to build a ``messages`` list from ``(role, content)`` tuples."""
    return [{"role": r, "content": c} for r, c in messages]


def make_greeting_sample(rng: random.Random) -> dict:
    user = rng.choice(["Hi", "Hello", "Hey", "Hi there", "Good morning",
                       "Hello!", "Hey there", "Hi!", "Greetings", "Hi friend",
                       "Hello again", "Morning!", "Evening", "Hiya", "Heyo"])
    answer = _wrap_assistant(rng, rng.choice(GREETING_RESPONSES))
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, user)),
            ("assistant", answer),
        ),
        "metadata": {"generator": "greeting"},
    }


def make_identity_sample(rng: random.Random) -> dict:
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, rng.choice(IDENTITY_QUESTIONS))),
            ("assistant", _wrap_assistant(rng, rng.choice(IDENTITY_ANSWERS))),
        ),
        "metadata": {"generator": "identity"},
    }


def make_capital_sample(rng: random.Random) -> dict:
    country, capital = rng.choice(COUNTRIES_CAPITALS)
    q_templates = [
        f"What is the capital of {country}?",
        f"Which city is the capital of {country}?",
        f"What's the capital city of {country}?",
        f"Tell me the capital of {country}.",
        f"{country} — what's its capital?",
    ]
    a_templates = [
        f"The capital of {country} is {capital}.",
        f"{capital} is the capital of {country}.",
        f"It's {capital}.",
        f"{capital} — that's the capital of {country}.",
    ]
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, rng.choice(q_templates))),
            ("assistant", _wrap_assistant(rng, rng.choice(a_templates))),
        ),
        "metadata": {"generator": "capital"},
    }


def make_animal_fact_sample(rng: random.Random) -> dict:
    animal, fact = rng.choice(ANIMAL_FACTS)
    q_templates = [
        f"How many legs does a {animal} have?",
        f"How many legs does an {animal} have?",
        f"How many legs does the {animal} have?",
        f"Tell me: how many legs does a {animal} have?",
    ]
    a_templates = [
        f"A {animal} has {fact}.",
        f"An {animal} has {fact}.",
        f"The {animal} has {fact}.",
    ]
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, rng.choice(q_templates))),
            ("assistant", _wrap_assistant(rng, rng.choice(a_templates))),
        ),
        "metadata": {"generator": "animal_fact"},
    }


def make_fact_sample(rng: random.Random) -> dict:
    q, a = rng.choice(SCIENCE_FACTS + GEOGRAPHY_FACTS + HISTORY_FACTS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, q)),
            ("assistant", _wrap_assistant(rng, a)),
        ),
        "metadata": {"generator": "fact"},
    }


def make_keyword_sample(rng: random.Random) -> dict:
    keyword, fact = rng.choice(list(KEYWORD_TO_FACT.items()))
    q = rng.choice([
        f"What is {keyword}?",
        f"Explain {keyword}.",
        f"Tell me about {keyword}.",
        f"What does {keyword} mean?",
        f"Could you describe {keyword}?",
    ])
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, q)),
            ("assistant", _wrap_assistant(rng, fact)),
        ),
        "metadata": {"generator": "keyword_explain"},
    }


def make_translate_sample(rng: random.Random) -> dict:
    eng, lang, trans = rng.choice(TRANSLATIONS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Translate '{eng}' to {lang}.")),
            ("assistant", _wrap_assistant(rng, f"'{eng}' in {lang} is '{trans}'.")),
        ),
        "metadata": {"generator": "translate"},
    }


def make_compound_translate_sample(rng: random.Random) -> dict:
    """Translate two word/language pairs in a single turn."""
    a = rng.choice(TRANSLATIONS)
    b = rng.choice(TRANSLATIONS)
    eng1, lang1, t1 = a
    eng2, lang2, t2 = b
    answer = f"'{eng1}' in {lang1} is '{t1}', and '{eng2}' in {lang2} is '{t2}'."
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Translate '{eng1}' to {lang1} and '{eng2}' to {lang2}.")),
            ("assistant", _wrap_assistant(rng, answer)),
        ),
        "metadata": {"generator": "translate_compound"},
    }


def make_summarize_sample(rng: random.Random) -> dict:
    src, summary = rng.choice(SUMMARIES)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Summarize this in one sentence: {src}")),
            ("assistant", _wrap_assistant(rng, summary)),
        ),
        "metadata": {"generator": "summarize"},
    }


def make_politer_rewrite_sample(rng: random.Random) -> dict:
    src, polite = rng.choice(POLITE_REWRITES)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Rewrite this politely: {src}")),
            ("assistant", _wrap_assistant(rng, polite)),
        ),
        "metadata": {"generator": "rewrite_polite"},
    }


def make_sentiment_sample(rng: random.Random) -> dict:
    label, text = rng.choice(SENTIMENT_TEXTS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Classify this as positive, neutral, or negative: {text}")),
            ("assistant", _wrap_assistant(rng, f"{label.capitalize()}.")),
        ),
        "metadata": {"generator": "sentiment"},
    }


def make_tips_sample(rng: random.Random) -> dict:
    topic, tip = rng.choice(TIPS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Give me three tips for {topic}.")),
            ("assistant", _wrap_assistant(rng, tip)),
        ),
        "metadata": {"generator": "tips"},
    }


def make_code_sample(rng: random.Random) -> dict:
    task, code = rng.choice(CODE_TASKS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Write a Python function that {task}.")),
            ("assistant", _wrap_assistant(rng, f"```python\n{code}\n```")),
        ),
        "metadata": {"generator": "code"},
    }


def make_math_sample(rng: random.Random) -> dict:
    """Random arithmetic problems from arithmetic generators."""
    kind = rng.choice(["add", "sub", "mul", "div", "mixed"])
    if kind == "add":
        a = rng.randint(1, 999)
        b = rng.randint(1, 999)
        expr = f"{a} + {b}"
        ans = str(a + b)
    elif kind == "sub":
        a = rng.randint(50, 999)
        b = rng.randint(1, a - 1)
        expr = f"{a} - {b}"
        ans = str(a - b)
    elif kind == "mul":
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        expr = f"{a} * {b}"
        ans = str(a * b)
    elif kind == "div":
        b = rng.randint(2, 25)
        ans_int = rng.randint(1, 50)
        a = b * ans_int
        expr = f"{a} / {b}"
        ans = str(ans_int)
    else:  # mixed
        a = rng.randint(2, 20)
        b = rng.randint(2, 20)
        c = rng.randint(2, 20)
        expr = f"({a} + {b}) * {c}"
        ans = str((a + b) * c)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Compute {expr}.")),
            ("assistant", _wrap_assistant(rng, f"{expr} = {ans}.")),
        ),
        "metadata": {"generator": "math"},
    }


def make_next_day_sample(rng: random.Random) -> dict:
    d = rng.choice(DAYS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"What comes after {d}?")),
            ("assistant", _wrap_assistant(rng, f"{NEXT_DAY[d]} comes after {d}.")),
        ),
        "metadata": {"generator": "next_day"},
    }


def make_month_days_sample(rng: random.Random) -> dict:
    month = rng.choice(MONTHS)
    days = MONTH_DAYS[month]
    if month == "February":
        suffix = " (28 in a common year, 29 in a leap year)"
    else:
        suffix = ""
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"How many days are in {month}?")),
            ("assistant", _wrap_assistant(rng, f"{month} has {days} days{suffix}.")),
        ),
        "metadata": {"generator": "month_days"},
    }


def make_book_recommendation_sample(rng: random.Random) -> dict:
    topic, ans = rng.choice(BOOKS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Recommend a book about {topic}.")),
            ("assistant", _wrap_assistant(rng, ans)),
        ),
        "metadata": {"generator": "book"},
    }


def make_haiku_sample(rng: random.Random) -> dict:
    topic, haiku = rng.choice(HAIKUS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Write a haiku about {topic}.")),
            ("assistant", _wrap_assistant(rng, haiku)),
        ),
        "metadata": {"generator": "haiku"},
    }


def make_joke_sample(rng: random.Random) -> dict:
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, rng.choice([
                "Tell me a joke.", "Got a joke?",
                "Make me laugh.", "Do you know any jokes?",
                "Tell me something funny.",
            ]))),
            ("assistant", _wrap_assistant(rng, rng.choice(JOKES))),
        ),
        "metadata": {"generator": "joke"},
    }


def make_refusal_sample(rng: random.Random) -> dict:
    q, a = rng.choice(REFUSALS)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, q)),
            ("assistant", _wrap_assistant(rng, a)),
        ),
        "metadata": {"generator": "refusal"},
    }


def make_followup_sample(rng: random.Random) -> dict:
    """Two-turn: question + follow-up."""
    base_q, base_a = rng.choice(SCIENCE_FACTS + GEOGRAPHY_FACTS + HISTORY_FACTS + [
        ("Tell me about Flatbuild.",
         "Flatbuild is an open-source framework for training compact conversational language models from scratch."),
        ("Tell me about Flatrun.",
         "Flatrun is a streaming inference runtime for language models in the Flat ecosystem."),
        ("What is a transformer model?",
         "A transformer is a neural network architecture that uses self-attention to process sequences."),
    ])
    follow_openers = [
        "Could you explain that more simply?",
        "What does that mean in practice?",
        "Can you give me an example?",
        "Why is that?",
        "Are you sure?",
        "What about edge cases?",
        "Any alternatives?",
        "Where can I read more?",
    ]
    follow_q = rng.choice(follow_openers)
    follow_ans = _wrap_assistant(rng, "Sure — " + base_a)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, base_q)),
            ("assistant", _wrap_assistant(rng, base_a)),
            ("user", _wrap_user(rng, follow_q)),
            ("assistant", follow_ans),
        ),
        "metadata": {"generator": "followup"},
    }


def make_context_switch_sample(rng: random.Random) -> dict:
    first_q, first_a = rng.choice(SCIENCE_FACTS + GEOGRAPHY_FACTS + HISTORY_FACTS)
    second_q, second_a = rng.choice(SCIENCE_FACTS + GEOGRAPHY_FACTS + HISTORY_FACTS + [
        ("Switch context: what is Flatbuild?",
         "Flatbuild is an open-source framework for training small conversational language models from scratch."),
    ])
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, first_q)),
            ("assistant", _wrap_assistant(rng, first_a)),
            ("user", _wrap_user(rng, second_q)),
            ("assistant", _wrap_assistant(rng, second_a)),
        ),
        "metadata": {"generator": "context_switch"},
    }


def make_three_turn_sample(rng: random.Random) -> dict:
    """Three user-assistant exchanges on related topics."""
    related = rng.sample(SCIENCE_FACTS + GEOGRAPHY_FACTS + HISTORY_FACTS, 3)
    messages = [("system", SYSTEM)]
    for q, a in related:
        messages.append(("user", _wrap_user(rng, q)))
        messages.append(("assistant", _wrap_assistant(rng, a)))
    return {"messages": _conv(*messages), "metadata": {"generator": "three_turn"}}


def make_counting_sample(rng: random.Random) -> dict:
    """'Count from N to M' style instructions."""
    a = rng.randint(1, 30)
    b = a + rng.randint(5, 100)
    seq = list(range(a, b + 1))
    fmt = ", ".join(str(s) for s in seq)
    return {
        "messages": _conv(
            ("system", SYSTEM),
            ("user", _wrap_user(rng, f"Count from {a} to {b}.")),
            ("assistant", _wrap_assistant(rng, f"{fmt}.")),
        ),
        "metadata": {"generator": "counting"},
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


GENERATOR_WEIGHTS: list[tuple[str, Callable[[random.Random], dict], int]] = [
    ("greeting",         make_greeting_sample,             7),
    ("identity",         make_identity_sample,             6),
    ("capital",          make_capital_sample,             10),
    ("animal_fact",      make_animal_fact_sample,          5),
    ("fact",             make_fact_sample,                15),
    ("keyword_explain",  make_keyword_sample,              4),
    ("translate",        make_translate_sample,            6),
    ("translate_compound", make_compound_translate_sample,  3),
    ("summarize",        make_summarize_sample,            3),
    ("rewrite_polite",   make_politer_rewrite_sample,      3),
    ("sentiment",        make_sentiment_sample,            4),
    ("tips",             make_tips_sample,                 4),
    ("code",             make_code_sample,                 6),
    ("math",             make_math_sample,                 7),
    ("next_day",         make_next_day_sample,             2),
    ("month_days",       make_month_days_sample,           2),
    ("counting",         make_counting_sample,             3),
    ("book",             make_book_recommendation_sample,  2),
    ("haiku",            make_haiku_sample,                2),
    ("joke",             make_joke_sample,                 2),
    ("refusal",          make_refusal_sample,              2),
    ("followup",         make_followup_sample,             2),
    ("context_switch",   make_context_switch_sample,       2),
    ("three_turn",       make_three_turn_sample,           2),
]


def generate(n: int, seed: int = 7) -> Iterator[dict]:
    """Yield ``n`` conversation samples."""
    rng = random.Random(seed)
    total = sum(w for _, _, w in GENERATOR_WEIGHTS)
    entries = list(GENERATOR_WEIGHTS)

    for i in range(n):
        pick = rng.uniform(0, total)
        cum = 0
        chosen = entries[-1]
        for entry in entries:
            cum += entry[2]
            if pick <= cum:
                chosen = entry
                break
        sample = chosen[1](rng)
        sample.setdefault("metadata", {})["sample_index"] = i
        yield sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo_large/dataset.jsonl", type=Path)
    parser.add_argument("--n", default=100_000, type=int)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for sample in generate(args.n, args.seed):
            counts[sample["metadata"]["generator"]] = counts.get(sample["metadata"]["generator"], 0) + 1
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n:,} samples to {args.out}")
    print(f"File size: {args.out.stat().st_size / 1e6:.1f} MB")
    print("Generator breakdown (top 10):")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    for k, v in top:
        print(f"  {k:>18}: {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
