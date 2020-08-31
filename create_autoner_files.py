import json
import spacy
from neo4j import GraphDatabase

nlp = spacy.load("en_core_web_sm")


def create_dict_core():
    # Connect to database, open session
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "test"))
    with driver.session() as session:
        # Get list of sensor types
        results = session.run('MATCH (s:Sensor) RETURN DISTINCT s.types')
        sensor_types_set = set()
        for sensor_types in results:
            types_list = sensor_types["s.types"]
            for sensor_type in types_list:
                if sensor_type not in sensor_types_set:
                    sensor_types_set.add(sensor_type)
        # Get list of sensor technologies
        results = session.run('MATCH (s:Sensor) RETURN DISTINCT s.technology')
        sensor_technologies_set = set()
        for sensor_technology_record in results:
            technology = sensor_technology_record["s.technology"]
            sensor_technologies_set.add(technology)
        # Get list of observable properties
        results = session.run('MATCH (o:ObservableProperty) RETURN DISTINCT o.name')
        observable_names_set = set()
        for result in results:
            name = result["o.name"]
            observable_names_set.add(name)

        # Get list of bands
        results = session.run('MATCH (s:Sensor) RETURN DISTINCT s.wavebands')
        bands_set = set()
        for band_result in results:
            bands_list = band_result["s.wavebands"]
            for band in bands_list:
                if band not in bands_set:
                    bands_set.add(band)

    with open("dict_core.txt", "w") as dict_core_file:
        for sensor_type in sensor_types_set:
            dict_core_file.write(f"Technology\t{sensor_type}\n")
        for sensor_tech in sensor_technologies_set:
            dict_core_file.write(f"Technology\t{sensor_tech}\n")
        for observable in observable_names_set:
            dict_core_file.write(f"Observable\t{observable}\n")
        for waveband in bands_set:
            dict_core_file.write(f"Waveband\t{waveband}\n")


def create_dict_full():
    phrases = []
    with open('AutoPhrase/models/CREATE/AutoPhrase.txt') as phrases_file:
        for phrase_info in phrases_file:
            phrase_array = phrase_info.split('\t')
            phrases.append(phrase_array[1])
    with open("dict_full.txt", "w") as dict_full_file:
        for phrase in phrases:
            dict_full_file.write(f"{phrase}")


def create_raw_text():
    with open('papers_test.json') as papers_json_file:
        papers_list = json.load(papers_json_file)
    with open('raw_text.txt', 'w', encoding='utf-8') as raw_text_file:
        for paper in papers_list:
            title_doc = nlp(paper["title"])
            for token in title_doc:
                raw_text_file.write(token.text + '\n')
            raw_text_file.write('\n')
            if "abstract" in paper:
                abstract_doc = nlp(paper["abstract"])
                for sentence in abstract_doc.sents:
                    for token in nlp(sentence.text):
                        raw_text_file.write(token.text + '\n')
                    raw_text_file.write('\n')


def create_autophrase_text():
    with open('papers_test.json') as papers_json_file:
        papers_list = json.load(papers_json_file)
    with open('AutoPhrase/data/input.txt', 'w', encoding='utf-8') as raw_text_file:
        for paper in papers_list:
            raw_text_file.write(paper["title"] + '\n')
            if "abstract" in paper:
                raw_text_file.write(paper["abstract"] + '\n')


if __name__ == '__main__':
    create_raw_text()
    create_autophrase_text()
    create_dict_core()
    create_dict_full()
