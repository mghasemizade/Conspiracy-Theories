from imports import *
#the dictionary of the labels from the family tree
ft = {"aviation": ["Chemtrails", "Black Helicopters", "Korean Airline 007", "Malaysia MH370", "Malaysia MH17"],
      "economics and business and society": ["NESARA", "Deepwater Horizon", "New Coke", "Illuminati", "New World Order", "Beyonce", "Denver Airport", "Whitney Houston", "George Soros", "Freemasonry", "Üst akıl{Master Mind}", "Hollywood is turning your kids gay", "Drug Trafficking", "McDonald"],
      "race and religion": ["Antisemitism", "Jewish Space Lasers", "Anti-Armenianism", "The Protocols of the Elders of Zion", "Anti-Baháʼísm", "Holocaust Denial", "Anti-Catholicism", "Rothschild Central Bank", "Antichrist", "Bible and Jesus", "White genocide", "Islam", "Black Genocide", "Great Replacement", "Racism", "Paul the apostle"],
      "government and politics": ["Paul McCartney", "Muhammadu Buhari", "Avril Lavigne", "Covered Up Deaths", "Anti-islami", "Death panel", "Loose Chain Films", "9/11", "Nero", "Deaths", "Sandy Hook", "Shootings", "Pizza Gate", "Espionage", "J.F. Kennedy", "Stolen Election", "Sutherland Springs", "Crisis Actors", "Parkland Shooting", "l", "QAnon", "Israel Animal Spying", "Clintons", "Harold Wilson", "Malala Yousafzi", "Jeffrey Epstein", "Russian interference", "Abraham Lincoln", "Marthin Luther king", "Eric V", "Dmitry Ivanovich", "Sheikh Rahman", "Yitzhak Rabin", "Zachary Taylor", "George S. Patton", "Diana Princess of Wales", "Dag Hammarskjöld", "Kurt Cobain", "Michael Jackson", "Marilyn Monroe", "Tupac Shakur", "Mozart", "John Lennon", "Jimi Hendrix", "Notorious B.I.G", "Pope John Paul I", "Jill Dando", "Olof Palme", "Chester Bennington", "Paul Walker", "David Kelly", "Shbash Chandra Bose", "Sushant Singh Rajput", "Elvis Presley", "Adolf Hilter", "Lord Lucan", "Madeleine McCann", "Seth Rich", "Federal Emergency Management Agency", "Wayfair", "African National Congress", "Italygate", "Barak Obama", "Satanic Ritual Abuse", "Cultural MArxism", "Adrenochrome", "Deep State", "Donald Trump", "Election-Ukraine-Biden", "Joe Biden", "Ukraine Gas Company", "Body Double", "Melania Trump", "False Flag Operations", "Biden", "Ukraine Bioweapons", "Malcom X", "October Surprise", "Nazis on the Moon", "Pedophile Priest", "Jimmy Savile", "Hurricane Maria", "Room 641a", "German Coup", "Jade Helm", "Buckingham Naked Boy"],
      "medical": ["Zika", "Chicken Farma", "HIV", "Flouridation", "Ebola", "ADHD", "Viruses", "Vaccines", "Covid-19", "Artifical Diseases", "HSV", "MMR", "Bill Gates as Scapegoat", "Cancer", "Alternative Therapy Suppression", "DPT", "HPV", "Small POX", "Polio", "Microchips"],
      "science and technology": ["Black Holes", "5G Phones", "MKUltra", "Earth", "HAARP", "Global Warming", "RFID", "Flat Earth", "Technlogoy Supression", "Climate Change", "Weaponry", "Hollow Earth", "Targeted Individuals", "False History"],
      "outer space": ["Area 51", "Moon Landing", "UFO", "Lost Cosmonauts", "12th Planet", "Men in Black", "Reptilians", "Dead Cattle", "Bush family, Margaret Thatcher, Bob Hope, and the British Royal Family", "Ancient Astronaut", "Alien Mummy", "Apollo 17"],
      "sports": ["Boxing", "Ali-Liston Fight", "Shregar Race-horse", "Bradley-Pacquiao", "Rigged Selection Processess", "Ronaldo 1998 World Cup", "New England Patriots"]}

def label_clusters_with_dict(df, ft, cluster_column='Cluster', text_column='text', dict_label_column='label2'):
    df['label2'] = ''  # Initialize the new column

    # Iterate over the dictionary items
    for key, values in ft.items():
        # Convert the values to lowercase for case-insensitive comparison
        values = [v.lower() for v in values]

        # Iterate over the rows of the data frame
        for i, row in df.iterrows():
            matches = []  # Store all matches here

            # Check if the text contains any of the values
            for value in values:
                if value in row[text_column].lower():
                    matches.append(key)

            # If there are matches, join them and assign to dict_label column
            if matches:
                df.at[i, dict_label_column] = ', '.join(matches)

    return df