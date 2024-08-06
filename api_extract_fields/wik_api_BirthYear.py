import json

import requests
from bs4 import BeautifulSoup

artist_info = {
    "אייל גולן": {"Year of Birth": 1971, "Music Style": "Pop"},
    "בנזין": {"Year of Birth": None, "Music Style": None},
    "דודו טסה": {"Year of Birth": None, "Music Style": None},
    "אגם בוחבוט": {"Year of Birth": None, "Music Style": None},
    "שלמה גרוניך": {"Year of Birth": None, "Music Style": None},
    "מתי כספי": {"Year of Birth": None, "Music Style": None},
    "שירי מימון": {"Year of Birth": None, "Music Style": None},
    "אבי ביטר": {"Year of Birth": None, "Music Style": None},
    "חמי רודנר": {"Year of Birth": None, "Music Style": None},
    "עומר אדם": {"Year of Birth": None, "Music Style": None},
    "יהודית רביץ": {"Year of Birth": None, "Music Style": None},
    "ישי לוי": {"Year of Birth": None, "Music Style": None},
    "טונה": {"Year of Birth": None, "Music Style": None},
    "הראל מויאל": {"Year of Birth": None, "Music Style": None},
    "אהובה עוזרי": {"Year of Birth": None, "Music Style": None},
    "שלום חנוך": {"Year of Birth": None, "Music Style": None},
    "מרגלית צנעני": {"Year of Birth": None, "Music Style": None},
    "יזהר אשדות": {"Year of Birth": None, "Music Style": None},
    "אריק ברמן": {"Year of Birth": None, "Music Style": None},
    "דני סנדרסון": {"Year of Birth": None, "Music Style": None},
    "נועה קירל": {"Year of Birth": None, "Music Style": None},
    "כוורת": {"Year of Birth": None, "Music Style": None},
    "עדן בן זקן": {"Year of Birth": None, "Music Style": None},
    "נונו": {"Year of Birth": None, "Music Style": None},
    "קובי פרץ": {"Year of Birth": None, "Music Style": None},
    "רן אלירן": {"Year of Birth": None, "Music Style": None},
    "עידן חביב": {"Year of Birth": None, "Music Style": None},
    "זוהר ארגוב": {"Year of Birth": None, "Music Style": None},
    "להקת פיקוד צפון": {"Year of Birth": None, "Music Style": None},
    "רמי פורטיס": {"Year of Birth": None, "Music Style": None},
    "אסתר עופרים": {"Year of Birth": None, "Music Style": None},
    "אביתר בנאי": {"Year of Birth": None, "Music Style": None},
    "גן חיות": {"Year of Birth": None, "Music Style": None},
    "התרנגולים": {"Year of Birth": None, "Music Style": None},
    "יוסי בנאי": {"Year of Birth": None, "Music Style": None},
    "סטפן לגר": {"Year of Birth": None, "Music Style": None},
    "יסמין מועלם": {"Year of Birth": None, "Music Style": None},
    "גלי עטרי": {"Year of Birth": None, "Music Style": None},
    "מני בגר": {"Year of Birth": None, "Music Style": None},
    "מוש בן ארי": {"Year of Birth": None, "Music Style": None},
    "פאר טסי": {"Year of Birth": None, "Music Style": None},
    "להקת פיקוד דרום": {"Year of Birth": None, "Music Style": None},
    "להקת חיל הים": {"Year of Birth": None, "Music Style": None},
    "סינרגיה": {"Year of Birth": None, "Music Style": None},
    "תיסלם": {"Year of Birth": None, "Music Style": None},
    "גבי שושן": {"Year of Birth": None, "Music Style": None},
    "ישי ריבו": {"Year of Birth": None, "Music Style": None},
    "ג'ירפות": {"Year of Birth": None, "Music Style": None},
    "הצל": {"Year of Birth": None, "Music Style": None},
    "שמעון בוסקילה": {"Year of Birth": None, "Music Style": None},
    "עטר מיינר": {"Year of Birth": None, "Music Style": None},
    "השמחות": {"Year of Birth": None, "Music Style": None},
    "אינפקציה": {"Year of Birth": None, "Music Style": None},
    "אהוד בנאי": {"Year of Birth": None, "Music Style": None},
    "להקת הנח\"ל": {"Year of Birth": None, "Music Style": None},
    "סי היימן": {"Year of Birth": None, "Music Style": None},
    "ליאור נרקיס": {"Year of Birth": None, "Music Style": None},
    "מוקי": {"Year of Birth": None, "Music Style": None},
    "רונה קינן": {"Year of Birth": None, "Music Style": None},
    "זהבה בן": {"Year of Birth": None, "Music Style": None},
    "עילי בוטנר": {"Year of Birth": None, "Music Style": None},
    "חוה אלברשטיין": {"Year of Birth": None, "Music Style": None},
    "אביב גפן": {"Year of Birth": None, "Music Style": None},
    "יהורם גאון": {"Year of Birth": None, "Music Style": None},
    "שפיות זמנית": {"Year of Birth": None, "Music Style": None},
    "דיקלה": {"Year of Birth": None, "Music Style": None},
    "יגאל בשן": {"Year of Birth": None, "Music Style": None},
    "הדג נחש": {"Year of Birth": None, "Music Style": None},
    "דניאל סלומון": {"Year of Birth": None, "Music Style": None},
    "אתי אנקרי": {"Year of Birth": None, "Music Style": None},
    "שלומי שבן": {"Year of Birth": None, "Music Style": None},
    "מאיר אריאל": {"Year of Birth": None, "Music Style": None},
    "בניה ברבי": {"Year of Birth": None, "Music Style": None},
    "שימי תבורי": {"Year of Birth": None, "Music Style": None},
    "שב\"ק ס'": {"Year of Birth": None, "Music Style": None},
    "סאבלימינל": {"Year of Birth": None, "Music Style": None},
    "יואב יצחק": {"Year of Birth": None, "Music Style": None},
    "אנה זק": {"Year of Birth": None, "Music Style": None},
    "דודו אהרון": {"Year of Birth": None, "Music Style": None},
    "עופרה חזה": {"Year of Birth": None, "Music Style": None},
    "גאולה גיל": {"Year of Birth": None, "Music Style": None},
    "יהודה פוליקר": {"Year of Birth": None, "Music Style": None},
    "היהודים": {"Year of Birth": None, "Music Style": None},
    "גידי גוב": {"Year of Birth": None, "Music Style": None},
    "אניה בוקשטיין": {"Year of Birth": None, "Music Style": None},
    "אריק לביא": {"Year of Birth": None, "Music Style": None},
    "עופר לוי": {"Year of Birth": None, "Music Style": None},
    "אלכסנדרה": {"Year of Birth": None, "Music Style": None},
    "אריק איינשטיין": {"Year of Birth": None, "Music Style": None},
    "היי פייב": {"Year of Birth": None, "Music Style": None},
    "עברי לידר": {"Year of Birth": None, "Music Style": None},
    "משה פרץ": {"Year of Birth": None, "Music Style": None},
    "רבקה זוהר": {"Year of Birth": None, "Music Style": None},
    "ברי סחרוף": {"Year of Birth": None, "Music Style": None},
    "אליעד נחום": {"Year of Birth": None, "Music Style": None},
    "אמיר דדון": {"Year of Birth": None, "Music Style": None},
    "מאיה בוסקילה": {"Year of Birth": None, "Music Style": None},
    "נינט טייב": {"Year of Birth": None, "Music Style": None},
    "אושיק לוי": {"Year of Birth": None, "Music Style": None},
    "אבנר גדסי": {"Year of Birth": None, "Music Style": None},
    "טיפקס": {"Year of Birth": None, "Music Style": None},
    "צלילי הכרם": {"Year of Birth": None, "Music Style": None},
    "חנן בן ארי": {"Year of Birth": None, "Music Style": None},
    "צביקה פיק": {"Year of Birth": None, "Music Style": None},
    "ארקדי דוכין": {"Year of Birth": None, "Music Style": None},
    "דור דניאל": {"Year of Birth": None, "Music Style": None},
    "שושנה דמארי": {"Year of Birth": None, "Music Style": None},
    "איתי לוי": {"Year of Birth": None, "Music Style": None},
    "החלונות הגבוהים": {"Year of Birth": None, "Music Style": None},
    "מירי מסיקה": {"Year of Birth": None, "Music Style": None},
    "קרן פלס": {"Year of Birth": None, "Music Style": None},
    "חיים משה": {"Year of Birth": None, "Music Style": None},
    "הפיל הכחול": {"Year of Birth": None, "Music Style": None},
    "ריטה": {"Year of Birth": None, "Music Style": None},
    "שלמה ארצי": {"Year of Birth": None, "Music Style": None},
    "יפה ירקוני": {"Year of Birth": None, "Music Style": None},
    "משינה": {"Year of Birth": None, "Music Style": None},
    "דורון מזר": {"Year of Birth": None, "Music Style": None},
    "כנסיית השכל": {"Year of Birth": None, "Music Style": None},
    "רוני": {"Year of Birth": None, "Music Style": None},
    "פבלו רוזנברג": {"Year of Birth": None, "Music Style": None},
    "מרינה מקסימיליאן בלומין": {"Year of Birth": None, "Music Style": None},
    "נורית גלרון": {"Year of Birth": None, "Music Style": None},
    "סטטיק ובן אל תבורי": {"Year of Birth": None, "Music Style": None},
    "בית הבובות": {"Year of Birth": None, "Music Style": None},
    "יוני רכטר": {"Year of Birth": None, "Music Style": None},
    "שרית חדד": {"Year of Birth": None, "Music Style": None},
    "גלעד שגב": {"Year of Birth": None, "Music Style": None},
    "אתניקס": {"Year of Birth": None, "Music Style": None},
    "מוניקה סקס": {"Year of Birth": None, "Music Style": None},
    "עמיר בניון": {"Year of Birth": None, "Music Style": None},
    "בועז שרעבי": {"Year of Birth": None, "Music Style": None},
    "דודי לוי": {"Year of Birth": None, "Music Style": None},
    "אחינועם ניני": {"Year of Birth": None, "Music Style": None},
    "דיאנה גולבי": {"Year of Birth": None, "Music Style": None},
    "נתן גושן": {"Year of Birth": None, "Music Style": None},
    "מאיר בנאי": {"Year of Birth": None, "Music Style": None},
    "אדם": {"Year of Birth": None, "Music Style": None},
    "אריק סיני": {"Year of Birth": None, "Music Style": None},
    "אסף אמדורסקי": {"Year of Birth": None, "Music Style": None},
    "עוזי חיטמן": {"Year of Birth": None, "Music Style": None},
    "אריאל זילבר": {"Year of Birth": None, "Music Style": None},
    "ניסים סרוסי": {"Year of Birth": None, "Music Style": None},
    "אברהם טל": {"Year of Birth": None, "Music Style": None},
    "יובל דיין": {"Year of Birth": None, "Music Style": None},
    "דנה אינטרנשיונל": {"Year of Birth": None, "Music Style": None},
    "רמי קלינשטיין": {"Year of Birth": None, "Music Style": None},
    "יהונתן מרגי": {"Year of Birth": None, "Music Style": None},
    "שלומי שבת": {"Year of Birth": None, "Music Style": None},
    "עידן עמדי": {"Year of Birth": None, "Music Style": None},
    "ריקי גל": {"Year of Birth": None, "Music Style": None},
    "שרי": {"Year of Birth": None, "Music Style": None},
    "דפנה ארמוני": {"Year of Birth": None, "Music Style": None},
    "התקווה 6": {"Year of Birth": None, "Music Style": None},
    "להקת חיל האויר": {"Year of Birth": None, "Music Style": None},
}
# Function to fetch artist information from Hebrew Wikipedia API

# Function to fetch artist information from Hebrew Wikipedia API
# Function to fetch artist information from Hebrew Wikipedia
def fetch_artist_info(artist_name):
    url = f"https://he.wikipedia.org/wiki/{artist_name}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        summary = soup.find('div', class_='mw-parser-output')
        import re

        if summary:
            summary_text = summary.get_text()
            if ('נולד' in summary_text) or ('נולדה' in summary_text):
                index = summary_text.find('נולד') if 'נולד' in summary_text else summary_text.find('נולדה')
                relevant_text = summary_text[index:index + 100]  # Adjust 23 to your desired range
                match = re.search(r'\b\d{4}\b', relevant_text)
                if match:
                    birth_year = int(match.group())
                    print(birth_year)
                    return birth_year
                else:
                    if (')' in summary_text) or (')' in summary_text):
                        index = summary_text.find(')') if '(' in summary_text else summary_text.find('(')
                        relevant_text = summary_text[index:index + 100]  # Adjust 23 to your desired range
                        match = re.search(r'\b\d{4}\b', relevant_text)
                        if match:
                            birth_year = int(match.group())
                            print(birth_year)
                            return birth_year
    else:
        print(f"Failed to fetch Wikipedia page for {artist_name}. Status code: {response.status_code}")

    return None


# Update the artist_info dictionary with birth years from Hebrew Wikipedia
for artist_name in artist_info.keys():
    if artist_info[artist_name]["Year of Birth"] is None:
        birth_year = fetch_artist_info(artist_name)
        artist_info[artist_name]["Year of Birth"] = birth_year

# Print the updated artist_info dictionary
for artist_name, info in artist_info.items():
    print(f"Artist: {artist_name}, Year of Birth: {info['Year of Birth']}, Music Style: {info['Music Style']}")

# Save the artist_info dictionary to a JSON file
# Save the dictionary to a JSON file
with open('../artist_info.json', 'w', encoding='utf-8') as json_file:
    json.dump(artist_info, json_file, ensure_ascii=False, indent=4)