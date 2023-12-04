from bs4 import BeautifulSoup
import requests
import string
import json

BASE_URL = "https://www.health.harvard.edu/a-through-c"
current_url = BASE_URL
letters = set()
dictionary = {}

def next_letter():
    for letter in string.ascii_lowercase:
        if letter not in letters:
            return letter
    return None

def push_letters(first_letter, last_letter):
    for i in range(ord(first_letter), ord(last_letter) + 1):
        letters.add(chr(i))


while True:
    # First requesting the webpage and parse it
    print("- Requesting and parsing: {}".format(current_url))
    re = requests.get(current_url)
    assert re.status_code == 200 # Ensure that the resource is not 

    # Find the letters using the url
    first_letter, last_letter = re.url.split("#")[0].split("/")[-1].split("-through-")
    push_letters(first_letter, last_letter)

    # Secondly parse the html content of the webpage and list all terms
    soup = BeautifulSoup(re.text, "html.parser")
    article = soup.find("div", class_="content-repository-content")
    link = None

    for term in article.find_all("p"):
        word = term.find("strong")
        if word is not None and word.string is not None:
            word = word.string.split(":")[0]
            definition = term.contents[-1]
            dictionary[word] = definition

        elif "Browse dictionary by letter:" in term.contents[0]:
            for link in term.find_all('a'):
                href = link.attrs['href']
                letter = link.string.replace(' ', '').lower()

                if not letter in letters:
                    link = href
                    break
    
    if link is None:
        break

    try:
        current_url = "https://www.health.harvard.edu" + link
    except Exception:
        break

# Finally export the dictionary
with open("medic-dictionary.json", "w") as f:
    f.write(json.dumps(dictionary, indent=4))
