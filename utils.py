import re
import requests
import wikipedia
from typing import List, Optional
from bs4 import BeautifulSoup
import openai
from openai import OpenAI

def extract_acronyms(text: str) -> List[str]:
    # Pattern for finding acronyms (2 or more uppercase letters)
    pattern = r'\b[A-Z]{2,}\b'
    acronyms = re.findall(pattern, text)
    return list(set(acronyms))  # Remove duplicates

def get_acronym_meanings(acronym: str, context_text: str) -> Optional[str]:
    try:

        # Step 1: Get meanings from abbreviations.com
        url = f"https://www.abbreviations.com/{acronym}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        found_meanings = []
        
        # Look for definitions in table cells and div classes
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            for i in range(len(cells)-1):
                if acronym in cells[i].get_text().strip():
                    desc = cells[i+1].find('p', class_='desc')
                    if desc:
                        meaning = desc.get_text().strip()
                        if meaning:
                            found_meanings.append(meaning)
        
        definition_divs = soup.find_all('div', class_='desc')
        for div in definition_divs:
            text = div.get_text().strip()
            if text:
                found_meanings.append(text)

        if not found_meanings:
            return None

        # Step 2: Use OpenAI to determine most relevant meaning based on context
        client = OpenAI()
        prompt = f"""Given the following context: '{context_text}'
        And these possible meanings for the acronym '{acronym}':
        {found_meanings}
        
        What is the most relevant meaning? Format your response exactly like this:
        FULL_FORM: <write the expanded form here>
        """

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content.strip()
        
        # Extract just the full form using regex
        full_form_match = re.search(r'FULL_FORM:\s*(.+)', response)
        most_relevant_meaning = full_form_match.group(1).strip() if full_form_match else response

        # Step 3: Search Wikipedia with the most relevant meaning
        wiki_results = wikipedia.search(most_relevant_meaning, results=5)
        
        if not wiki_results:
            return None

        # Step 4: Use OpenAI to determine most relevant Wikipedia page
        wiki_summaries = []
        for result in wiki_results:
            try:
                page = wikipedia.page(result, auto_suggest=False)
                wiki_summaries.append({
                    "title": page.title,
                    "summary": page.summary.split('\n')[0]
                })
            except:
                continue
        
        if not wiki_summaries:
            return None

        prompt = f"""Given the acronym '{acronym}' and its meaning '{most_relevant_meaning}',
        which of these Wikipedia articles is most relevant? 
        
        {'\n'.join(f"{i+1}. {s['title']}: {s['summary']}" for i, s in enumerate(wiki_summaries))}
        
        Format your response exactly like this:
        INDEX: <number>"""

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract index using regex
        response = completion.choices[0].message.content.strip()
        index_match = re.search(r'INDEX:\s*(\d+)', response)
        if not index_match:
            return None
        most_relevant_index = int(index_match.group(1)) - 1
        
        # Step 5: Return the summary of the most relevant page
        return wiki_summaries[most_relevant_index]["summary"]

    except Exception as e:
        print(f"Error processing {acronym}: {str(e)}")
        return None
