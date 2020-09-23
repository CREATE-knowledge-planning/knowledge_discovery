import sqlite3


def create_db_structure():
    conn = sqlite3.connect('phrases.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE phrases
                 (phrase text, type text)''')
    conn.commit()
    conn.close()


def load_phrase_list():
    conn = sqlite3.connect('phrases.db')
    c = conn.cursor()
    with open("AutoPhrase.txt") as raw_phrases_file:
        for line in raw_phrases_file:
            prob, phrase = line.split('\t')
            c.execute("INSERT INTO phrases VALUES(?, ?)", (phrase[:-1], "TBD"))
    conn.commit()
    conn.close()


def classify_phrases():
    conn = sqlite3.connect('phrases.db')
    c = conn.cursor()
    for row in c.execute("SELECT * FROM phrases WHERE type='TBD'"):
        phrase = row[0]
        print(phrase)
        input_type = input("Please indicate type for this entity (1: Measurement; 2: Sensor Technology; 3: Waveband, 4: None; 5: TBD, 6: Stop this system)")
        selected_type = None
        if input_type == "1":
            selected_type = "Observable"
        elif input_type == "2":
            selected_type = "Technology"
        elif input_type == "3":
            selected_type = "Waveband"
        elif input_type == "4":
            selected_type = "Ignore"
        elif input_type == "5":
            selected_type = "TBD"
        elif input_type == "6":
            break
        if selected_type is not None:
            upd_cursor = conn.cursor()
            upd_cursor.execute("UPDATE phrases SET type=? WHERE phrase = ?", (selected_type, phrase))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    classify_phrases()
