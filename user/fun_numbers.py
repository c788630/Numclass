"""
fun_numbers.py

This category contains fun numbers famous in sci-fi, internet memes,
pop culture, geek subcultures and computing.
Add, edit, or remove entries as you like!

Keys must be integers.
"""

FUN_NUMBERS = {
    7:      "Considered a lucky or sacred number in many cultures and religions, often exaggerated "
            "in memes and fiction. '7 of 9' is a Star Trek: Voyager character (not literal, but often "
            "referenced numerically).",
    13:     "13—unlucky, also the number of colonies in Battlestar Galactica.",
    23:     "23 enigma (Illuminatus! Trilogy, Lost, pop culture mysticism).",
    42:     "The Answer to the Ultimate Question of Life, the Universe, and Everything "
			"(Douglas Adams, The Hitchhiker’s Guide to the Galaxy).",
    47:     "Star Trek writers' in-joke: number 47 appears frequently in the franchise.",
    54:     "Studio 54 (pop culture/celebrity hub).",
    69:     "Pop culture meme number (humor, innuendo).",
    88:     "88 miles per hour—the speed needed for time travel (Back to the Future)",
    99:     "Agent 99 (Get Smart, classic spy comedy).",
    101:    "Room 101 (Orwell's 1984); also 'Intro 101' as a code for basics in US education.",
    112:    "The emergency telephone number throughout the European Union and many other countries "
            "worldwide; equivalent to 911 in the US/Canada.",
    221:    "221B Baker Street (Sherlock Holmes, referenced in Doctor Who and elsewhere).",
    314:    "Pi (3.14), beloved in science/math pop culture.",
    327:    "Docking Bay 327 in Star Wars: A New Hope (Millennium Falcon arrival).",
	404:    "A common meme for 'not found' errors or missing content.",
	420:    "Internet meme and counterculture reference (esp. 4:20 PM as 'weed time').",
    451:    "the title of 'Fahrenheit 451', Ray Bradbury’s novel, named for the temperature at which book paper ignites.",
	666:    "The Number of the Beast: a beastly (or hateful) number from the Bible (Revelation); "
            "popular in pop culture, heavy metal, horror, and edgy memes (see also: OEIS A051003).",
    911:    "The emergency telephone number in the US, Canada and several other countries; "
            "also associated with the September 11, 2001 terrorist attacks in the US.",
    1138:   "THX 1138 (George Lucas's first feature film and recurring Star Wars/Lucasfilm Easter egg).",
    1337:   "'Leet' speak for 'elite', iconic in hacker/gamer culture.",
    1701:   "NCC-1701—registry number of the Starship Enterprise (Star Trek).",
    1984:   "Nineteen Eighty-Four, classic dystopian novel (George Orwell).",
    2001:   "2001 is the year featured in '2001: A Space Odyssey' by Arthur C. Clarke and Stanley Kubrick.",
    2112:   "Rush concept album (sci-fi dystopia, cult favorite in geek culture).",
    2187:   "Leia's cell number in Star Wars: A New Hope (cell 2187); also an homage to experimental film '21-87'",
	3141:   "The first four digits of pi (π = 3.141...) — math and pop culture symbol.",
	9000:   "HAL 9000, sentient computer (2001: A Space Odyssey).",
	9001:   "It's Over 9000!—internet meme from Dragon Ball Z.",
	13337:  "A Leet number: 'a Leeter' (playful extension of leet).",
    24601:  "Jean Valjean's prisoner number (Les Misérables); geek culture references.",
    31337:  "A Leet number: 'Eleet' or 'elite'; highly regarded in old-school hacker circles.",
    32767:  "The maximum value for a signed 16-bit integer (2¹⁵–1); often seen as a classic "
            "data limit (e.g., MIDI, audio, legacy software).",
   -32768:  "The minimum value for a signed 16-bit integer (–2¹⁵); underflows/overflows in "
            "legacy software and hardware.",
    65535:  "The maximum value for an unsigned 16-bit integer (2¹⁶–1); common as the highest "
            " port number (TCP/UDP), color depth, and memory limits in classic computing.",
    74656:  "Registry of USS Voyager (Star Trek: Voyager).",
    80085:  'A calculator gag spelling of “BOOBS” upside-down.',
    123456: "A joke about weak passwords.",
    5318008: 'A calculator gag spelling "BOOBIES" upside-down.',
    8675309: "Jenny's phone number (Tommy Tutone, iconic in pop culture).",
    13371337: "A Leet number: 'Double leet'; playful combination of leet digits.",
    16777215: "The maximum value for a 24-bit integer (2²⁴–1); classic color value in "
              "true color (24-bit, 16.7 million colors: #FFFFFF).",
    133769420: 'A Leet giggle; it combines meme numbers 1337, 69, and 420.',
    1234567890: "UNIX time for 13 Feb 2009 23:31:30 GMT (computer/geek humor).",
    2147483647: "The maximum value for a signed 32-bit integer; the largest representable Unix time "
                "(seconds since Jan 1, 1970) in 32-bit systems. Will cause the 'Year 2038 problem' "
                " or 'Unix Millennium Bug' when the counter overflows.",
    4294967295: "The maximum value for an unsigned 32-bit integer (2³²–1); appears in programming, "
                "file size limits, IP address space (255.255.255.255), and more.",
    281474976710655: "The maximum value for a 48-bit integer (2⁴⁸–1); used in MAC addresses and "
                     "classic computing boundaries.",
    18446744073709551615: "The maximum value for an unsigned 64-bit integer (2⁶⁴–1); "
                          "'all bits set' in 64-bit computing.",
}